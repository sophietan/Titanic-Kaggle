import csv
from enum import Enum
from typing import List, Dict, Any
from tabulate import tabulate


def load_data(file: str, is_train) -> List[Dict[str, Any]]:
    global header
    with open(file) as f:
        raw_train = csv.reader(f)
        header = next(raw_train)
        data = [dict(zip(header, row)) for row in raw_train]
    for row in data:
        for attribute, type in attr_types.items():
            value = row[attribute]
            # Check if predicted value
            if not is_train and attribute == 'Survived':
                continue
            # Assign enums to enum values
            if issubclass(type, Enum):
                for enums in type:
                    if value == enums.value:
                        row[attribute] = enums
                        break
                else:
                    raise ValueError(f'Value {value} not in {type.__name__}')
                continue

            if attribute == 'Name':
                names.add(value)
            # Cabin magic
            if attribute == 'Cabin':
                cabins.add(value)
                if len(row[attribute]) == 3:
                    row['Deck'] = row[attribute][0]

                else:
                    row['Deck'] = ''

                continue
            try:
                row[attribute] = type(value)

            except ValueError:
                row[attribute] = None
    return data


class Embarked(Enum):
    Unknown = ''
    Cherbourg = 'C'
    Queenstown = 'Q'
    Southampton = 'S'


class Sex(Enum):
    Female = 'female'
    Male = 'male'


attr_types = {
    'PassengerId': int,
    'Pclass': int,
    'Name': str,
    'Sex': Sex,
    'Age': int,
    'SibSp': int,
    'Parch': int,
    'Ticket': int,
    'Fare': float,
    'Cabin': str,
    'Embarked': Embarked
}

cabins = set()
deck = set()
names = set()
titles = set()


train = load_data('train.csv', is_train=True)
test = load_data('test.csv', is_train=False)

def survived(data) -> int:
    return len([p for p in train if p['Survived']])

no_survived = survived(train)

def add_extra_features(data: List) -> List:
    for row in data:
        row['Family Size'] = row['Parch'] + row['SibSp'] + 1
        if row['Fare'] is None:
            row['Fare per person'] = None
        else:
            row['Fare per person'] = row['Fare'] / row['Family Size']

        Name = row['Name']
        First_Names = Name.split(', ', maxsplit=1)[1].strip()
        Title = First_Names.split('. ', maxsplit=1)[0]
        titles.add(Title)
        row['Title'] = Title

    return data

def hist(attr: str):
    hist = {}
    for row in data:
        if row[attr] not in hist:
            hist[row[attr]] = [0, 0]
        hist[row[attr]][row['Survived'] == '1'] += 1

    return hist

def value_count(attr: str, relative=True):
    possible_attribute_values = set()
    row_map = {}
    data = []

    # {attribute: X, test: Y, train: Z}

    for row in train:
        possible_attribute_values.add(row[attr])

    for row in test:
        possible_attribute_values.add(row[attr])

    for n, value in enumerate(possible_attribute_values):
        row_map[value] = n
        data.append({attr: value, 'TR S': 0, 'TR D': 0, 'TR': 0, 'TE': 0})

    for row in train:
        data_row = data[row_map[row[attr]]]
        data_row['TR'] += 1
        if row['Survived'] == '1':
            data_row['TR S'] += 1
        else:
            data_row['TR D'] += 1

    for row in test:
        data_row = data[row_map[row[attr]]]
        data_row['TE'] += 1

    table(data)
    return data

train = add_extra_features(train)
test = add_extra_features(test)

def table(data):
    transformed = []
    headers = list(data[0].keys())
    for row in data:
        entry = []
        for h in headers:
            entry.append(row[h])
        transformed.append(entry)
    print(tabulate(transformed, headers=headers))

