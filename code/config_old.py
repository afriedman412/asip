

EPSILON = 0.01

EMOTION_MODELS = [
    "michellejieli/emotion_text_classifier",
    # "mrm8488/t5-base-finetuned-emotion",
    "cardiffnlp/twitter-roberta-base-emotion",
    "j-hartmann/emotion-english-distilroberta-base",
    "cardiffnlp/twitter-roberta-base-emotion-latest"
]

TOXICITY_MODEL = "unitary/toxic-bert"

TOPIC_MODEL = "facebook/bart-large-mnli"


MAIN_CHARS = {
    'asip': [
        'CHARLIE', 'DENNIS', 'FRANK', 'MAC', 'SWEET DEE'
    ],
    'office': [
        'andy', 'angela', 'darryl', 'dwight', 'erin', 'jim', 'kevin', 'michael',
        'oscar', 'pam', 'ryan'
    ],
    'sp': [
        'cartman', 'kyle', 'stan', 'randy', 'butters', 'sharon', 'stephen',
        'wendy', 'mr. mackey', 'jimmy'
    ]

}

ALL_CHARS = MAIN_CHARS['asip'] + MAIN_CHARS['office'] + MAIN_CHARS['sp']

SHOWS = ['asip', 'office', 'southpark']

EMOTIONS = [
    'anger',
    'anticipation',
    'disgust',
    'fear',
    'joy',
    'love',
    'optimism',
    'pessimism',
    'sadness',
    'surprise',
    'trust'
]

EMO_COLS = [
    'anger_mjm',
    'disgust_mjm',
    'fear_mjm',
    'joy_mjm',
    'sadness_mjm',
    'surprise_mjm',
    'anger_cardiff',
    'joy_cardiff',
    'optimism_cardiff',
    'sadness_cardiff',
    'anger_hartmann',
    'disgust_hartmann',
    'fear_hartmann',
    'joy_hartmann',
    'sadness_hartmann',
    'surprise_hartmann',
    'anger_new',
    'anticipation_new',
    'disgust_new',
    'fear_new',
    'joy_new',
    'love_new',
    'optimism_new',
    'pessimism_new',
    'sadness_new',
    'surprise_new',
    'trust_new'
]


TOXIC_COLS = [

    'identity_hate_toxic', 'insult_toxic', 'obscene_toxic', 'severe_toxic',
    'threat_toxic', 'toxic_toxic'
]

TOPICS = [
    "race and ethnicity",
    "LGBTQ issues",
    "gun policy and firearms",
    "religion and faith",
    "electoral politics and government",
    "socioeconomic class and inequality",
    "illegal drugs and substance policy",
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
