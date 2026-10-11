//! The signal of a mark (`Client::mark`) on wgpu.

use crate::compute::device_poison::PoisonWatch;
use cubecl_core::server::{CompletionSignal, ServerError};
use cubecl_environment::future::DynFut;
use cubecl_environment::future::channel::{self, Receiver, Sender};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A point on the queue. The stream completes it from the
/// `on_submitted_work_done` callback of the first submission after the mark,
/// which covers every submission before it, because the queue completes its
/// submissions in order.
#[derive(Debug)]
pub(crate) struct WgpuSignal {
    done: AtomicBool,
    /// Closed on completion, or by the poison watch when the device is lost,
    /// which wakes every waiter at once.
    sender: Sender<()>,
    receiver: Receiver<()>,
    poison: PoisonWatch,
}

impl WgpuSignal {
    pub(crate) fn new(poison: PoisonWatch) -> Arc<Self> {
        let (sender, receiver) = channel::bounded(1);
        poison.wake_on_loss(&sender);
        Arc::new(Self {
            done: AtomicBool::new(false),
            sender,
            receiver,
            poison,
        })
    }

    /// The device has completed the work before the mark.
    pub(crate) fn complete(&self) {
        self.done.store(true, Ordering::Release);
        self.sender.close();
    }
}

impl CompletionSignal for WgpuSignal {
    fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire) || self.poison.is_poisoned()
    }

    fn wait(self: Arc<Self>) -> DynFut<Result<(), ServerError>> {
        Box::pin(async move {
            if !self.done.load(Ordering::Acquire) {
                // Only a close wakes this: nothing sends.
                let _ = self.receiver.recv().await;
            }
            self.poison.check()
        })
    }
}
