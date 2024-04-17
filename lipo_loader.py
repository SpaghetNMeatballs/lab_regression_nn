from lipo_dataclass import LipoParams
import pandas as pd


def load_lipo() -> tuple[LipoParams, ...]:
    lipo_dataframe = pd.read_csv("lipo.csv")
    result = []
    for i in lipo_dataframe.iterrows():
        row_content = i[1]
        result.append(
            LipoParams(
                name=row_content[0],
                logP=row_content[1],
                parameters=pd.Series.tolist(row_content[2:]),
            )
        )
    return tuple(result)


if __name__ == "__main__":
    load_lipo()
