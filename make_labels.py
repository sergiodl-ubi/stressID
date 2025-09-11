import argparse
from enum import Enum
from typing import override
import pandas as pd
from dataclasses import dataclass


@dataclass
class ProgramParameters:
    input: str
    output: str | None


parser = argparse.ArgumentParser(
    prog="Built_Labels",
    description="Creates a labels.csv file from StressID self_assessments.csv file.",
)
_ = parser.add_argument(
    "-i", "--input", help='Emotional Self-Assessment data CSV filename. default: "./self_assessments.csv"'
)
_ = parser.add_argument("-o", "--output", help='Output csv file to store every task labels. default: "./labels.csv"')
args = parser.parse_args(namespace=ProgramParameters)

input_file = args.input if args.input else "./self_assessments.csv"
output_file = args.output if args.output else "./labels.csv"


class BinaryClasses(Enum):
    NoStressed = 0
    Stressed = 1


class TernaryClasses(Enum):
    Relaxed = 0
    Stressed = 1
    RealStress = 2


class QuaternaryClasses(Enum):
    Relaxed = 0
    Stressed = 1
    RealStress = 2
    Amused = 3


class Task:
    task: str = ""
    stressed: float = 0
    relaxed: float = 0
    valence: float | None = 0
    arousal: float | None = 0
    binaryClass: BinaryClasses = BinaryClasses.NoStressed
    ternaryClass: TernaryClasses = TernaryClasses.Relaxed
    quaternaryClass: QuaternaryClasses = QuaternaryClasses.Relaxed

    def __init__(self, task: str, data: dict[str, str]) -> None:
        self.task = task
        self.stressed = float(data["S"])
        self.relaxed = float(data["R"])
        self.valence = None if data["V"] == "" else float(data["V"])
        self.arousal = None if data["A"] == "" else float(data["A"])
        self.binaryClassify()
        self.ternaryClassify()
        self.quaternaryClassify()

    def binaryClassify(self) -> None:
        self.binaryClass = BinaryClasses.Stressed if self.stressed >= 5 else BinaryClasses.NoStressed

    def ternaryClassify(self) -> None:
        isStressed = self.stressed >= 5
        isRelaxed = self.relaxed >= 5
        isAroused = self.arousal >= 5 if self.arousal else True
        isNotAroused = self.arousal <= 5 if self.arousal else True
        positiveValence = self.valence >= 5 if self.valence else True
        negativeValence = self.valence <= 5 if self.valence else True
        clearStress = (self.stressed - self.relaxed) >= 3
        realStress = (isStressed and clearStress) or ((self.stressed - self.relaxed) >= 5)
        clearRelax = (self.relaxed - self.stressed) >= 3
        realRelax = (isRelaxed and clearRelax) or ((self.relaxed - self.stressed) >= 5)

        if (isStressed and isAroused and (negativeValence or clearStress)) or realStress:
            self.ternaryClass = TernaryClasses.RealStress
        elif (isRelaxed and (isNotAroused or clearRelax) and positiveValence) or realRelax:
            self.ternaryClass = TernaryClasses.Relaxed
        else:
            self.ternaryClass = TernaryClasses.Stressed

    def quaternaryClassify(self) -> None:
        # Based on "Affect Representation and Recognition in 3D Continuous Valence-Arousal-Dominance Space"
        # Verma, Gyanendra & Tiwary, Uma Shanker. Multimedia Tools and Applications. 2017
        # DOI: 10.1007/s11042-015-3119-y
        # Stressed and relaxed reported levels follow StressID methodology
        if (
            self.stressed >= 5
            and (self.arousal >= 5 if self.arousal else True)
            and (self.valence <= 5 if self.valence else True)
        ):
            self.quaternaryClass = QuaternaryClasses.RealStress
        elif (
            (self.relaxed > 5 or ((self.relaxed - self.stressed) > 3))
            and (self.arousal <= 5 if self.arousal else True)
            and (self.valence >= 5 if self.valence else True)
        ):
            self.quaternaryClass = QuaternaryClasses.Relaxed
        elif (
            self.stressed < 5
            and (self.arousal >= 4 if self.arousal else True)
            and (self.valence >= 5 if self.valence else True)
        ):
            self.quaternaryClass = QuaternaryClasses.Amused
        else:
            self.quaternaryClass = QuaternaryClasses.Stressed

    @override
    def __str__(self) -> str:
        return (
            f"Task({self.task}) "
            + f"B({self.binaryClass.name}[{self.binaryClass.value}]) T({self.ternaryClass.name}[{self.ternaryClass.value}]) Q({self.quaternaryClass.name}[{self.quaternaryClass.value}]) - "
            + f"[Stressed: {self.stressed}, Relaxed: {self.relaxed}] "
            + f"[Valence: {self.valence}, Arousal: {self.arousal}"
        )


def row_to_tasks(index: str, row_data: dict[str, str]) -> dict[str, dict[str, str]]:
    result = {
        f"{index}_Breathing": {
            "R": row_data["Breathing_relax"],
            "S": row_data["Breathing_stress"],
            "V": row_data["Breathing_valence"],
            "A": row_data["Breathing_arousal"],
        },
        f"{index}_Counting1": {
            "R": row_data["Counting1_relax"],
            "S": row_data["Counting1_stress"],
            "V": row_data["Counting1_valence"],
            "A": row_data["Counting1_arousal"],
        },
        f"{index}_Counting2": {
            "R": row_data["Counting2_relax"],
            "S": row_data["Counting2_stress"],
            "V": row_data["Counting2_valence"],
            "A": row_data["Counting2_arousal"],
        },
        f"{index}_Counting3": {
            "R": row_data["Counting3_relax"],
            "S": row_data["Counting3_stress"],
            "V": row_data["Counting3_valence"],
            "A": row_data["Counting3_arousal"],
        },
        f"{index}_Math": {
            "R": row_data["Math_relax"],
            "S": row_data["Math_stress"],
            "V": row_data["Math_valence"],
            "A": row_data["Math_arousal"],
        },
        f"{index}_Reading": {
            "R": row_data["Reading_relax"],
            "S": row_data["Reading_stress"],
            "V": row_data["Reading_valence"],
            "A": row_data["Reading_arousal"],
        },
        f"{index}_Relax": {
            "R": row_data["Relax_relax"],
            "S": row_data["Relax_stress"],
            "V": row_data["Relax_valence"],
            "A": row_data["Relax_arousal"],
        },
        f"{index}_Speaking": {
            "R": row_data["Speaking_relax"],
            "S": row_data["Speaking_stress"],
            "V": row_data["Speaking_valence"],
            "A": row_data["Speaking_arousal"],
        },
        f"{index}_Stroop": {
            "R": row_data["Stroop_relax"],
            "S": row_data["Stroop_stress"],
            "V": row_data["Stroop_valence"],
            "A": row_data["Stroop_arousal"],
        },
        f"{index}_Video1": {
            "R": row_data["Video1_relax"],
            "S": row_data["Video1_stress"],
            "V": row_data["Video1_valence"],
            "A": row_data["Video1_arousal"],
        },
        f"{index}_Video2": {
            "R": row_data["Video2_relax"],
            "S": row_data["Video2_stress"],
            "V": row_data["Video2_valence"],
            "A": row_data["Video2_arousal"],
        },
    }
    return result


labels: dict[str, list[str | float]] = {
    "subject/task": [],
    "binary-stress": [],
    "affect3-class": [],
    "affect4-class": [],
}

if __name__ == "__main__":
    reported_data = pd.read_csv(input_file, sep=";", index_col=0, dtype=str, keep_default_na=False)  # pyright: ignore[reportUnknownMemberType]

    for subjID, data in reported_data.iterrows():
        tasks = row_to_tasks(subjID, data)  # pyright: ignore[reportArgumentType]
        for taskName, values in tasks.items():
            if values["R"] == "" or values["S"] == "":
                continue
            task = Task(task=taskName, data=values)
            print(task)
            labels["subject/task"].append(task.task)
            labels["binary-stress"].append(task.binaryClass.value)
            labels["affect3-class"].append(task.ternaryClass.value)
            labels["affect4-class"].append(task.quaternaryClass.value)

    df = pd.DataFrame(labels)
    df.sort_values(by="subject/task", inplace=True)
    df.to_csv(output_file, sep=",", index=False)
