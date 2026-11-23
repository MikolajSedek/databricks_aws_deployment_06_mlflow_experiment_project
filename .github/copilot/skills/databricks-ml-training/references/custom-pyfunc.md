# Custom PyFunc Models from Code (`mlflow.pyfunc`)

When custom preprocessing, business-tier post-processing, or multi-model ensembling is needed alongside an XGBoost booster, use the **Models from Code** pattern (`python_model="path/to/model.py"` + `mlflow.models.set_model()`).

```python
import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from xgboost import Booster


class ChurnRiskPyFuncModel(PythonModel):
    """Custom PyFunc wrapper adding risk-tier classification around an XGBoost Booster."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.booster = Booster()
        self.booster.load_model(context.artifacts["xgb_booster"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(model_input)
        probs = self.booster.predict(dmatrix)
        return pd.DataFrame(
            {
                "churn_probability": probs,
                "risk_tier": ["HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW" for p in probs],
            }
        )
```
