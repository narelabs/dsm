from dsm import DynamicSegmentedMemory


memory = DynamicSegmentedMemory(".dsm/example.json", active_segment_limit=3)

memory.write(
    "Rust websocket services need bounded channels, ping timeouts and cancellation-safe tasks.",
    category_path="Programming → Rust → Async",
    importance=0.9,
)
memory.write(
    "MemoryOS and HMT organize agent memory into short, medium and long-term layers.",
    category_path="Science → AI → Memory",
    importance=0.8,
)
memory.write(
    "A personal preference can be stored as a high-priority long-term segment.",
    category_path="Personal → Preferences",
    importance=0.7,
)

context = memory.active_context("Как исправить websocket timeout в Rust?", k=2)
print(context.context_text)
memory.save()
