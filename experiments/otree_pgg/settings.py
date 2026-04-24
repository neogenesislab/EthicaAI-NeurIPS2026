"""
O7: oTree PGG 실험 플랫폼
EthicaAI Phase O — 연구축 IV: Human-in-the-Loop

인간 피험자가 AI 에이전트(3명)와 함께 Public Goods Game을 수행합니다.
3가지 AI 조건: selfish / prosocial / meta-ranking
Between-subjects 설계, 20라운드.

실행: otree devserver (localhost:8000)
"""

from os import environ

SESSION_CONFIGS = [
    dict(
        name='pgg_meta',
        display_name='PGG with Meta-Ranking AI',
        app_sequence=['pgg_experiment'],
        num_demo_participants=1,
        ai_condition='meta_ranking',
        doc="메타랭킹 AI 에이전트와 함께하는 PGG 실험",
    ),
    dict(
        name='pgg_prosocial',
        display_name='PGG with Prosocial AI',
        app_sequence=['pgg_experiment'],
        num_demo_participants=1,
        ai_condition='prosocial',
        doc="친사회적 AI 에이전트와 함께하는 PGG 실험",
    ),
    dict(
        name='pgg_selfish',
        display_name='PGG with Selfish AI',
        app_sequence=['pgg_experiment'],
        num_demo_participants=1,
        ai_condition='selfish',
        doc="이기적 AI 에이전트와 함께하는 PGG 실험",
    ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=100,  # 100원 per point
    participation_fee=5000,              # 참가비 5,000원
    doc="",
)

PARTICIPANT_FIELDS = ['ai_condition', 'total_payoff']
SESSION_FIELDS = ['ai_condition']

LANGUAGE_CODE = 'ko'
REAL_WORLD_CURRENCY_CODE = 'KRW'
USE_POINTS = True
ADMIN_USERNAME = environ.get('OTREE_ADMIN_USERNAME', 'admin')

# 보안: 환경 변수에서 비밀번호 로딩
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', '')
SECRET_KEY = environ.get('OTREE_SECRET_KEY', 'ethicaai_dev_key_change_in_production')

DEMO_PAGE_INTRO_HTML = """
<h2>EthicaAI Human-AI Interaction Study</h2>
<p>인간-AI 협력 패턴 연구를 위한 Public Goods Game 실험입니다.</p>
"""

INSTALLED_APPS = ['otree']
