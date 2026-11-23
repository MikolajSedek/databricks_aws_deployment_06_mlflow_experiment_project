# Declarative Feature Views (`databricks-feature-engineering>=0.16.0`)

Declarative Feature Views define time-windowed aggregations (`SlidingWindow`, `TumblingWindow`, `RollingWindow`) or latest value selections (`ColumnSelection`) over a `DeltaTableSource` or `StreamSource`.

```python
from datetime import timedelta
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    AggregationFunction,
    Avg,
    DeltaTableSource,
    Feature,
    OfflineStoreConfig,
    CronSchedule,
    TumblingWindow,
)

fe = FeatureEngineeringClient()
source = DeltaTableSource(
    catalog_name="data_engineering_dev",
    schema_name="gold",
    table_name="fct_orders",
)

avg_order_30d = Feature(
    source=source,
    entity=["sk_user"],
    timeseries_column="_ldts",
    function=AggregationFunction(
        operator=Avg(input="net_amount"),
        time_window=TumblingWindow(window_duration=timedelta(days=30)),
    ),
    name="user_avg_net_amount_30d",
)
```
