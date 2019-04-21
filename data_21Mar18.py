import pandas as pd
from enum import Enum


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

categories = train.columns.values

def simplify_name(df):
    titles = []
    first_names = []
    for name in df['Name']:
        label = name.split(', ', maxsplit=1)[1].strip()
        title = label.split('. ', maxsplit=1)[0]
        first_name = label.split(title + '.', maxsplit=1)[1].strip()
        titles.append(title)
        first_names.append(first_name)
    df['Titles'] = titles
    df['First Name'] = first_names
    return df

def calc_family_size(df):
    df['Family Size'] = df['SibSp'] + df['Parch'] + 1
    return df

def define_age_bracket(df):
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student',
                   'Young Adult', 'Adult', 'Senior']
    brackets = pd.cut(df.Age, bins, labels=group_names)
    df['Age Bracket'] = brackets
    return df

def process_age(df):
    mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean)
    df = define_age_bracket(df)
    return df

def process_fare(df):
    mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(mean)
    df['Fare Quartile'] = pd.qcut(df['Fare'], [0, .25, .5, .75, 1.],
                                    labels=['1_quartile', '2_quartile',
                                            '3_quartile', '4_quartile'])
    return df

def process_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Deck'] = [cabin[0] for cabin in df['Cabin']]
    return df

def process_data(df):
    df = process_age(df)
    df = calc_family_size(df)
    df = process_fare(df)
    df = process_cabin(df)
    df = simplify_name(df)
    return df

print(process_data(train.sample(5)))

