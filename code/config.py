

EPSILON = 0.01

EMOTION_MODELS = [
    "michellejieli/emotion_text_classifier",
    "mrm8488/t5-base-finetuned-emotion",
    "cardiffnlp/twitter-roberta-base-emotion",
    "j-hartmann/emotion-english-distilroberta-base",
    "cardiffnlp/twitter-roberta-base-emotion-latest"  # added later
]

TOXICITY_MODEL = "unitary/toxic-bert"


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

# can't remember if this works
ALL_CHARS = MAIN_CHARS['asip'] + MAIN_CHARS['office'] + MAIN_CHARS['sp']

EMOTIONS = [
    'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'optimism'
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
EMOTION_ORDER = ['anger', 'disgust', 'fear',
                 'joy', 'optimism', 'sadness', 'surprise']
TOPIC_ORDER = ['LGBTQ issues', 'race and ethnicity', 'religion and faith']
SHOW_ORDER = ['asip', 'office', 'southpark']

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

EMOTION_PALETTE = {
    'anger':    '#d62728',
    'disgust':  '#7a9a01',
    'fear':     '#6a3d9a',
    'joy':      '#ffb703',
    'optimism': '#2a9d8f',
    'sadness':  '#1f77b4',
    'surprise': '#e377c2',
}
