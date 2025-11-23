

EPSILON = 0.01

EMOTION_MODELS = [
    "michellejieli/emotion_text_classifier",          # MJM
    "cardiffnlp/twitter-roberta-base-emotion",        # CDF0
    "j-hartmann/emotion-english-distilroberta-base",  # HART
    "cardiffnlp/twitter-roberta-base-emotion-latest"  # CDF1
]

TOXICITY_MODEL = "unitary/toxic-bert"

TOPIC_MODEL = "facebook/bart-large-mnli"

MAIN_CHARS = {
    'ASIP': [
        'CHARLIE', 'DENNIS', 'FRANK', 'MAC', 'SWEET_DEE'
    ],
    'OFFICE': [
        'ANDY', 'ANGELA', 'DARRYL', 'DWIGHT', 'ERIN', 'JIM',
        'KEVIN', 'MICHAEL', 'OSCAR', 'PAM', 'RYAN'
    ],
    'SP': [
        'CARTMAN', 'KYLE', 'STAN',
        'RANDY', 'BUTTERS', 'SHARON', 'STEPHEN', 'WENDY', 'MR._MACKEY', 'JIMMY'
    ]

}

ALL_CHARS = MAIN_CHARS['ASIP'] + MAIN_CHARS['OFFICE'] + MAIN_CHARS['SP']

SHOWS = list(MAIN_CHARS.keys())

EMOTIONS = [
    'ANGER',
    'ANTICIPATION',
    'DISGUST',
    'FEAR',
    'JOY',
    'LOVE',
    'NEUTRAL',
    'OPTIMISM',
    'PESSIMISM',
    'SADNESS',
    'SURPRISE',
    'TRUST'
]


EMO_COLS = [
    'ANGER_MJM', 'DISGUST_MJM', 'FEAR_MJM', 'JOY_MJM', 'NEUTRAL_MJM',
    'SADNESS_MJM', 'SURPRISE_MJM', 'ANGER_CDF0', 'JOY_CDF0',
    'OPTIMISM_CDF0', 'SADNESS_CDF0', 'ANGER_HART', 'DISGUST_HART',
    'FEAR_HART', 'JOY_HART', 'NEUTRAL_HART', 'SADNESS_HART',
    'SURPRISE_HART', 'ANGER_CDF1', 'ANTICIPATION_CDF1', 'DISGUST_CDF1',
    'FEAR_CDF1', 'JOY_CDF1', 'LOVE_CDF1', 'OPTIMISM_CDF1', 'PESSIMISM_CDF1',
    'SADNESS_CDF1', 'SURPRISE_CDF1', 'TRUST_CDF1',
]


TOXIC_COLS = [
    'IDENTITY_HATE', 'INSULT', 'OBSCENE', 'SEVERE_TOXIC', 'THREAT', 'TOXIC'
]

TOPICS_FULL = [
    'SOCIOECONOMIC_CLASS_AND_INEQUALITY', 'RACE_AND_ETHNICITY',
    'ELECTORAL_POLITICS_AND_GOVERNMENT', 'RELIGION_AND_FAITH',
    'LGBTQ_ISSUES', 'GUN_POLICY_AND_FIREARMS',
    'ILLEGAL_DRUGS_AND_SUBSTANCE_POLICY'
]

TOPICS = [
    'CLASS', 'RACE', 'POLITICS', 'RELIGION',
    'GAYS', 'GUNS', 'DRUGS'
]

ALL_METRICS = TOPICS + TOXIC_COLS + EMO_COLS

# FILE ADDRESS
SP_LINES = "sp_char_counts_111225.csv"
SP_DF = "sp_all_data_111215.csv"
SP_DF_TOPICS = "south_park_scene_topics_111225.csv"
PREPRO = "merged_data_for_modeling_103025.csv"
NEW_CARDIFF = "new_cardiff_results_111525.csv"

# PLOTTING HELPERS
# EMOTION_ORDER = [
#     'anger',
#     'anticipation',
#     'disgust',
#     'fear',
#     'joy',
#     'love',
#     'optimism',
#     'pessimism',
#     'sadness',
#     'surprise',
#     'trust'
#     ]
TOPIC_ORDER = ['LGBTQ issues', 'race and ethnicity', 'religion and faith']


TOPIC_PALETTE = {
    'LGBTQ issues':       '#2a9d8f',
    'race and ethnicity': '#f4a261',
    'religion and faith': '#9d4edd',
}

LEGEND_LABELS = {
    'LGBTQ issues':       'LGBTQ',
    'race and ethnicity': 'Race',
    'religion and faith': 'Religion',
}

SHOW_LABELS = {
    'asip': "Always Sunny",
    "office": "The Office",
    "southpark": "South Park"
}

EMOTION_PALETTE = {
    'anger':        '#d62728',   # red
    'disgust':      '#7a9a01',   # olive green
    'fear':         '#6a3d9a',   # deep purple
    'joy':          '#ffb703',   # warm yellow
    'optimism':     '#2a9d8f',   # teal
    'sadness':      '#1f77b4',   # classic blue
    'surprise':     '#e377c2',   # pink/magenta
    'anticipation': '#f17c0f',   # orange (fits between joy + anger)
    'love':         '#ff6b6b',   # warm coral (distinct from anger red)
    'pessimism':    '#4b4b4b',   # charcoal grey (semantic match)
    # clean green (balanced, distinct from disgust)
    'trust':        '#4caf50',
}
