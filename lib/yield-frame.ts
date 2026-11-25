// Cooperative yield to the browser event loop. MessageChannel tasks are not
// subject to the 4ms nested-setTimeout clamp (same technique as React's
// scheduler), so frequent time-sliced yields stay cheap while still letting
// the browser paint and handle input between slices.

let channel: MessageChannel | null = null;
const queue: Array<() => void> = [];

export function yieldToMain(): Promise<void> {
  if (typeof MessageChannel === 'undefined') {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  if (!channel) {
    channel = new MessageChannel();
    channel.port1.onmessage = () => {
      const next = queue.shift();
      next?.();
    };
  }
  return new Promise((resolve) => {
    queue.push(resolve);
    channel!.port2.postMessage(null);
  });
}
