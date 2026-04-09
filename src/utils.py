BASE_FEATURES = [
    'height', 'age', 'appearance', 'goals', 'assists',
    'yellow cards', 'red cards', 'minutes played',
    'days_injured', 'games_injured', 'award',
    'position_encoded', 'winger',
    'value_drop_ratio', 'injury_burden', 'productivity_score'
]

GK_EXTRA_FEATURES = ['goals conceded', 'clean sheets']

CLUSTER_FEATURES = [
    'age', 'goals', 'assists', 'appearance', 'minutes played',
    'days_injured', 'yellow cards', 'red cards', 'award',
    'productivity_score', 'injury_burden', 'position_encoded'
]

INJURY_FEATURES = [
    'age', 'height', 'appearance', 'minutes played',
    'days_injured', 'games_injured', 'position_encoded',
    'yellow cards', 'red cards', 'injury_burden'
]

TARGET_VALUE = 'current_value'
TARGET_DEPRECIATION = 'value_drop_ratio'
TARGET_INJURY = 'injury_risk_label'
