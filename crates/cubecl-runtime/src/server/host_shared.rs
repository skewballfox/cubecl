//! What a host that streams input into a live buffer needs: a token that says
//! when the device is done with work queued so far, and a view that writes
//! into the buffer without a copy where the memory permits it.

use super::{Handle, ServerError};
use crate::client::Client;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::any::Any;
use core::ptr::NonNull;
use cubecl_common::bytes::Bytes;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{AtomicBool, Ordering};

/// When [`Server::mark`](super::Server::mark) gives the marked work to the device.
///
/// A runtime that sends each launch to the device at once (CUDA, HIP, the CPU
/// runtime) treats both the same.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MarkSubmit {
    /// Submit the queued work now. The token completes without any other
    /// call, at the cost of one submission for each mark.
    #[default]
    Now,
    /// Leave the queued work where it is, and let the next flush of the stream
    /// submit it. Launches stay batched, but the token does not complete until
    /// something flushes the stream: a later launch that fills the batch, a
    /// read, a sync, or [`Client::wait`] on the token.
    Deferred,
}

/// A point on a stream that the device signals when it has completed every
/// task queued before the point. A runtime gives one from
/// [`Server::mark`](super::Server::mark).
pub trait CompletionSignal: Send + Sync + core::fmt::Debug {
    /// Whether the device has reached the point. Does not block.
    fn is_done(&self) -> bool;

    /// Resolves when the device has reached the point.
    ///
    /// # Errors
    ///
    /// The fault the device reports, when the stream itself failed.
    fn wait(self: Arc<Self>) -> DynFut<Result<(), ServerError>>;
}

/// A point on a stream, from [`Client::mark`]. Ask it with
/// [`is_done`](Self::is_done), or wait for it with [`Client::wait`].
///
/// A host that writes a buffer which earlier launches read keeps the token of
/// the last of those launches, and writes again only when the token is done.
#[derive(Clone, Debug)]
pub struct CompletionToken {
    state: TokenState,
    stream_id: StreamId,
    submit: MarkSubmit,
}

#[derive(Clone, Debug)]
enum TokenState {
    /// The runtime signals the point itself.
    Signal(Arc<dyn CompletionSignal>),
    /// The runtime has no signal: the point is done after a sync of its
    /// stream that [`Client::wait`] starts.
    AfterSync(Arc<AtomicBool>),
    /// The mark was refused; [`Client::wait`] reports why.
    Failed(ServerError),
}

impl CompletionToken {
    pub(crate) fn new(
        signal: Result<Option<Arc<dyn CompletionSignal>>, ServerError>,
        stream_id: StreamId,
        submit: MarkSubmit,
    ) -> Self {
        let state = match signal {
            Ok(Some(signal)) => TokenState::Signal(signal),
            Ok(None) => TokenState::AfterSync(Arc::new(AtomicBool::new(false))),
            Err(err) => TokenState::Failed(err),
        };
        Self {
            state,
            stream_id,
            submit,
        }
    }

    /// Whether the device has completed every task queued before the mark.
    /// Does not block.
    ///
    /// On a runtime without a signal of its own, this is `true` only after
    /// [`Client::wait`] on this token. A refused mark is done: the wait
    /// reports the refusal.
    #[must_use]
    pub fn is_done(&self) -> bool {
        match &self.state {
            TokenState::Signal(signal) => signal.is_done(),
            TokenState::AfterSync(done) => done.load(Ordering::Acquire),
            TokenState::Failed(_) => true,
        }
    }

    /// The stream the mark is on.
    #[must_use]
    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    pub(crate) fn wait(&self, client: &Client) -> DynFut<Result<(), ServerError>> {
        match &self.state {
            TokenState::Signal(signal) => {
                if self.submit == MarkSubmit::Deferred
                    && let Err(err) = client.flush_stream(self.stream_id)
                {
                    return Box::pin(core::future::ready(Err(err)));
                }
                signal.clone().wait()
            }
            TokenState::AfterSync(done) => {
                let sync = client.sync_stream(self.stream_id);
                let done = done.clone();
                Box::pin(async move {
                    sync.await?;
                    done.store(true, Ordering::Release);
                    Ok(())
                })
            }
            TokenState::Failed(err) => Box::pin(core::future::ready(Err(err.clone()))),
        }
    }
}

/// Host-visible memory of a live allocation, from
/// [`Server::host_view`](super::Server::host_view).
pub struct MappedMemory {
    ptr: NonNull<u8>,
    len: usize,
    /// Keeps the allocation bound, so the pool neither gives the range to
    /// another handle nor relocates it while the view exists.
    _keep: Box<dyn Any + Send>,
}

impl MappedMemory {
    /// Memory at `ptr` of `len` bytes, valid while `keep` lives.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads and writes of `len` bytes from the host
    /// for as long as `keep` lives, and nothing else may hold a Rust reference
    /// to that memory in that time.
    #[must_use]
    pub unsafe fn new(ptr: NonNull<u8>, len: usize, keep: Box<dyn Any + Send>) -> Self {
        Self {
            ptr,
            len,
            _keep: keep,
        }
    }
}

// SAFETY: the memory belongs to the view alone (see `MappedMemory::new`), and
// `_keep` is `Send`.
unsafe impl Send for MappedMemory {}

impl core::fmt::Debug for MappedMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MappedMemory")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// A writable host view of a handle, from [`Client::host_view_mut`].
///
/// A mapped view writes into the allocation itself. A staged view is a host
/// buffer that starts zeroed, and that the drop of the view writes to the
/// handle with [`Client::write`].
pub struct HostView {
    inner: ViewInner,
}

enum ViewInner {
    Mapped(MappedMemory),
    Staged {
        data: Vec<u8>,
        handle: Handle,
        client: Client,
    },
}

impl HostView {
    pub(crate) fn mapped(memory: MappedMemory) -> Self {
        Self {
            inner: ViewInner::Mapped(memory),
        }
    }

    pub(crate) fn staged(handle: Handle, client: Client) -> Self {
        let len = usize::try_from(handle.size_in_used()).expect("a handle fits in host memory");
        Self {
            inner: ViewInner::Staged {
                data: vec![0; len],
                handle,
                client,
            },
        }
    }

    /// Whether the view writes into the allocation itself, with no copy.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        matches!(self.inner, ViewInner::Mapped(_))
    }
}

impl core::ops::Deref for HostView {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.inner {
            // SAFETY: valid for `len` bytes while the view lives (see `MappedMemory::new`).
            ViewInner::Mapped(memory) => unsafe {
                core::slice::from_raw_parts(memory.ptr.as_ptr(), memory.len)
            },
            ViewInner::Staged { data, .. } => data,
        }
    }
}

impl core::ops::DerefMut for HostView {
    fn deref_mut(&mut self) -> &mut [u8] {
        match &mut self.inner {
            // SAFETY: valid for `len` bytes while the view lives, and `&mut self`
            // makes this the only reference (see `MappedMemory::new`).
            ViewInner::Mapped(memory) => unsafe {
                core::slice::from_raw_parts_mut(memory.ptr.as_ptr(), memory.len)
            },
            ViewInner::Staged { data, .. } => data,
        }
    }
}

impl Drop for HostView {
    fn drop(&mut self) {
        if let ViewInner::Staged {
            data,
            handle,
            client,
        } = &mut self.inner
        {
            let data = core::mem::take(data);
            client.write(handle, Bytes::from_bytes_vec(data));
        }
    }
}

impl core::fmt::Debug for HostView {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostView")
            .field("mapped", &self.is_mapped())
            .field("len", &self.len())
            .finish()
    }
}
