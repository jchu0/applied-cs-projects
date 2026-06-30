# Distributed Streaming Analytics

Flink-like stream processing framework with windowing, checkpointing, and exactly-once semantics.

## Features

- **DataStream API**: Fluent API for stream transformations
- **Windowing**: Tumbling, sliding, session windows
- **Stateful Processing**: Managed state with checkpointing
- **Exactly-Once**: Guaranteed processing semantics
- **Event Time**: Watermarks and late data handling

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from streamanalytics import StreamExecutionEnvironment, TimeWindow

# Create execution environment
env = StreamExecutionEnvironment.get_execution_environment()

# Define streaming pipeline
stream = (
    env.from_source(kafka_source)
    .map(parse_event)
    .filter(lambda x: x.value > 0)
    .key_by(lambda x: x.user_id)
    .window(TimeWindow.tumbling(seconds=60))
    .aggregate(sum_values)
    .sink(output_sink)
)

# Execute
env.execute("My Streaming Job")
```

## Windowing

```python
from streamanalytics import TimeWindow, CountWindow

# Time-based windows
stream.window(TimeWindow.tumbling(minutes=5))
stream.window(TimeWindow.sliding(minutes=10, slide=minutes(1)))
stream.window(TimeWindow.session(gap=minutes(30)))

# Count-based windows
stream.window(CountWindow.tumbling(100))
```

## Stateful Processing

```python
from streamanalytics import KeyedProcessFunction

class CountWithState(KeyedProcessFunction):
    def open(self, ctx):
        self.count = ctx.get_state("count", default=0)

    def process_element(self, value, ctx):
        self.count += 1
        ctx.update_state("count", self.count)
        yield (ctx.current_key, self.count)
```

## Checkpointing

```python
env.enable_checkpointing(interval_ms=60000)
env.set_checkpoint_storage("file:///checkpoints")
```

## Testing

```bash
pytest tests/ -v  # 208 tests
```
