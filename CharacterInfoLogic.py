import pandas as pd
import json
import model.CharacterInfo as characterInfo
import random
import base64

df = pd.read_csv('CharacterData.csv', header=0)
classesList = df["Character"].tolist()


def get_info(characterClass):
    print(characterClass)
    characterInfoRow = df[(df["Class"] == characterClass)]
    print(characterInfoRow)

    info = characterInfo.CharacterInfo(str(characterInfoRow.iloc[0]["Character"]))
    return json.dumps(info.__dict__)


def get_suggestion():
    suggestion = random.choice(classesList)
    return suggestion
