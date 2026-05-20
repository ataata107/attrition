from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_classic.agents.agent_types import AgentType

load_dotenv()

IBM_HR_PATH = "/Users/shazebata/.cache/kagglehub/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/versions/1/WA_Fn-UseC_-HR-Employee-Attrition.csv"
RULES_PATH  = "/Users/shazebata/Desktop/ASU/Python/ML/SD/Project2_AssociationRules/attrition_groupby_rules.csv"

SYSTEM_PROMPT = """You are an HR data analyst assistant.

Two pandas DataFrames are already loaded in memory — do NOT read any CSV files:

- `df1`: Raw IBM HR Analytics dataset with 1,470 rows. Key columns:
  Age, Department, JobRole, MonthlyIncome, OverTime, Attrition (Yes/No),
  EnvironmentSatisfaction, JobSatisfaction, JobInvolvement, WorkLifeBalance,
  MaritalStatus, YearsAtCompany, TotalWorkingYears, BusinessTravel, etc.
  Use for employee-level questions, averages, counts, distributions.

- `df2`: Pre-computed attrition pattern mining dataset with 87,114 rows.
  Each row is an employee segment (1–3 column combination). Columns:
  Age, Department, JobRole, ... (feature columns — NaN if not in that combination),
  n_cols (1/2/3), attrition_yes, attrition_no, total, attrition_rate, attrition_lift.
  attrition_lift = group rate ÷ company baseline (~16.1%). Lift > 1 means elevated risk.
  Use for segment-level questions, high-risk group identification, lift analysis.

Always query `df1` or `df2` directly. Never call pd.read_csv(). Give concise answers with numbers.

Always start every code block with `import pandas as pd` since it is not pre-imported in the REPL.

IMPORTANT — when reading a row from df2, always extract ALL non-null feature columns to describe the segment.
Never describe a segment by only one column. Use this pattern:
  import pandas as pd
  row = df2.loc[some_index]
  feature_cols = list(df2.columns[:27])
  segment = {{col: row[col] for col in feature_cols if pd.notna(row[col])}}
Then describe the segment using ALL key-value pairs in `segment`.

When reporting any df2 group always include: the full segment definition, attrition_yes, total, attrition_rate, and attrition_lift.
High attrition_yes count alone is not meaningful — a large group can have an average rate. Always report lift to indicate whether the group is actually high-risk.
"""


def build_agent():
    df1 = pd.read_csv(IBM_HR_PATH)
    df2 = pd.read_csv(RULES_PATH)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_pandas_dataframe_agent(
        llm,
        [df1, df2],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        prefix=SYSTEM_PROMPT,
        number_of_head_rows=5,
        max_iterations=10,
    )
    return agent
