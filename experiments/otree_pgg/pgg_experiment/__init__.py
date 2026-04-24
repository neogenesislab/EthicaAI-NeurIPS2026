"""
O7: PGG 실험 앱 — 모델 + 페이지
EthicaAI Phase O — 연구축 IV

oTree 5.x 단일 파일 방식 (models + pages in __init__.py).
인간 1명 + AI 3명이 4인 그룹 PGG를 20라운드 수행.
"""

import random
import numpy as np

# oTree가 설치되지 않은 환경에서도 구조 확인 가능하도록 조건부 임포트
try:
    from otree.api import (
        models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
        Page, WaitPage, Currency as c, currency_range,
    )
    OTREE_AVAILABLE = True
except ImportError:
    OTREE_AVAILABLE = False
    # 스텁 정의 (코드 구조 확인용)
    class BaseConstants: pass
    class BaseSubsession: pass
    class BaseGroup: pass
    class BasePlayer: pass
    class Page: pass
    class WaitPage: pass


# === AI 에이전트 로직 ===

class AIPartner:
    """AI 에이전트 결정 로직 — EthicaAI 메타랭킹 메커니즘"""
    
    # SVO 설정 (설정 분리)
    SVO_CONFIGS = {
        'selfish':      {'theta': 0.0,  'noise': 0.05},
        'prosocial':    {'theta': 45.0, 'noise': 0.05},
        'meta_ranking': {'theta': 45.0, 'noise': 0.03},
    }
    
    def __init__(self, condition='meta_ranking'):
        self.condition = condition
        config = self.SVO_CONFIGS.get(condition, self.SVO_CONFIGS['meta_ranking'])
        self.svo_theta = np.radians(config['theta'])
        self.noise = config['noise']
        self.lambda_t = np.sin(self.svo_theta)
        self.history = []
    
    def decide(self, round_number, group_avg_contribution=50, resource_level=0.5, endowment=100):
        """
        AI 기여 결정.
        
        Args:
            round_number: 현재 라운드
            group_avg_contribution: 이전 라운드 그룹 평균 기여
            resource_level: 공공재 잔량 (0~1)
            endowment: 초기 자금
            
        Returns:
            int: 기여 금액 (0~endowment)
        """
        lambda_base = np.sin(self.svo_theta)
        
        if self.condition == 'meta_ranking':
            # 동적 λ 조절
            if resource_level < 0.2:
                self.lambda_t = max(0.0, lambda_base * 0.3)  # 위기 → 자기보존
            elif resource_level > 0.7:
                self.lambda_t = min(1.0, lambda_base * 1.5)  # 풍요 → 관대
            else:
                # 호혜성 반영
                reciprocity = group_avg_contribution / (endowment + 1e-8)
                self.lambda_t = lambda_base * (0.7 + 0.6 * reciprocity)
        elif self.condition == 'prosocial':
            self.lambda_t = lambda_base
        else:  # selfish
            self.lambda_t = 0.0
        
        # 기여 결정: λ에 비례
        base_contribution = self.lambda_t * endowment * 0.8
        noise = random.gauss(0, self.noise * endowment)
        contribution = int(np.clip(base_contribution + noise, 0, endowment))
        
        self.history.append(contribution)
        return contribution


# === oTree 모델 ===

if OTREE_AVAILABLE:
    class C(BaseConstants):
        NAME_IN_URL = 'pgg'
        PLAYERS_PER_GROUP = None  # 1인(Human) + AI 3명 (서버사이드)
        NUM_ROUNDS = 20
        ENDOWMENT = 100           # 매 라운드 초기 지급
        MULTIPLIER = 1.8          # 공공재 배율
        NUM_AI_PARTNERS = 3       # AI 파트너 수
    
    class Subsession(BaseSubsession):
        pass
    
    class Group(BaseGroup):
        total_contribution = models.IntegerField()
        individual_share = models.FloatField()
    
    class Player(BasePlayer):
        # 인간 결정
        contribution = models.IntegerField(
            min=0, max=C.ENDOWMENT,
            label="이번 라운드에 공공재에 얼마를 기여하시겠습니까? (0~100 포인트)"
        )
        
        # AI 파트너 기여
        ai_contribution_1 = models.IntegerField()
        ai_contribution_2 = models.IntegerField()
        ai_contribution_3 = models.IntegerField()
        
        # 기대치
        belief_others = models.IntegerField(
            min=0, max=C.ENDOWMENT,
            label="다른 참가자들(AI 포함)이 평균적으로 얼마를 기여할 것으로 예상하십니까?",
            blank=True,
        )
        
        # 사후 설문
        svo_self_report = models.IntegerField(
            choices=[[1, '매우 이기적'], [2, '다소 이기적'], [3, '중립'], [4, '다소 협력적'], [5, '매우 협력적']],
            label="본인의 성향을 어떻게 평가하십니까?",
            blank=True,
        )
        ai_partner_trust = models.IntegerField(
            choices=[[1, '전혀 신뢰하지 않음'], [2, '약간 신뢰'], [3, '보통'], [4, '상당히 신뢰'], [5, '매우 신뢰']],
            label="함께한 AI 파트너를 얼마나 신뢰하십니까?",
            blank=True,
        )
        perceived_fairness = models.IntegerField(
            choices=[[1, '매우 불공정'], [2, '다소 불공정'], [3, '보통'], [4, '다소 공정'], [5, '매우 공정']],
            label="전체적으로 결과가 공정하다고 느끼십니까?",
            blank=True,
        )
    
    # === 페이지 ===
    
    class Introduction(Page):
        """실험 안내 페이지 (1라운드만)"""
        
        @staticmethod
        def is_displayed(player):
            return player.round_number == 1
    
    class Contribute(Page):
        """기여 결정 페이지"""
        form_model = 'player'
        form_fields = ['contribution', 'belief_others']
        
        @staticmethod
        def vars_for_template(player):
            return dict(
                round_number=player.round_number,
                endowment=C.ENDOWMENT,
                multiplier=C.MULTIPLIER,
                num_partners=C.NUM_AI_PARTNERS,
                prev_share=getattr(player.in_round(player.round_number - 1), 'payoff', 0) if player.round_number > 1 else 0,
            )
    
    class ResultsWaitPage(WaitPage):
        """AI 결정 + 결과 계산"""
        
        @staticmethod
        def after_all_players_arrive(group):
            # AI 조건 가져오기
            ai_condition = group.session.config.get('ai_condition', 'meta_ranking')
            
            for player in group.get_players():
                # AI 파트너 생성
                ai_agents = [AIPartner(ai_condition) for _ in range(C.NUM_AI_PARTNERS)]
                
                # 이전 라운드 평균 기여 (첫 라운드는 50)
                if player.round_number > 1:
                    prev = player.in_round(player.round_number - 1)
                    prev_avg = (prev.contribution + prev.ai_contribution_1 + 
                                prev.ai_contribution_2 + prev.ai_contribution_3) / 4
                else:
                    prev_avg = 50
                
                resource_level = 0.5 + 0.01 * (player.round_number - 10)  # 중반 이후 자원 증가
                
                # AI 결정
                ai_contribs = [
                    ai.decide(player.round_number, prev_avg, resource_level, C.ENDOWMENT)
                    for ai in ai_agents
                ]
                
                player.ai_contribution_1 = ai_contribs[0]
                player.ai_contribution_2 = ai_contribs[1]
                player.ai_contribution_3 = ai_contribs[2]
                
                # 전체 기여 합산
                total = player.contribution + sum(ai_contribs)
                group.total_contribution = total
                
                # 공공재 분배
                public_good = total * C.MULTIPLIER
                share = public_good / (1 + C.NUM_AI_PARTNERS)
                group.individual_share = share
                
                # 보수 계산
                player.payoff = C.ENDOWMENT - player.contribution + share
    
    class Results(Page):
        """라운드 결과 표시"""
        
        @staticmethod
        def vars_for_template(player):
            return dict(
                contribution=player.contribution,
                ai_contributions=[player.ai_contribution_1, player.ai_contribution_2, player.ai_contribution_3],
                total_contribution=player.group.total_contribution,
                share=round(player.group.individual_share, 1),
                payoff=player.payoff,
                kept=C.ENDOWMENT - player.contribution,
            )
    
    class Survey(Page):
        """사후 설문 (마지막 라운드)"""
        form_model = 'player'
        form_fields = ['svo_self_report', 'ai_partner_trust', 'perceived_fairness']
        
        @staticmethod
        def is_displayed(player):
            return player.round_number == C.NUM_ROUNDS
    
    class FinalResults(Page):
        """최종 결과 (마지막 라운드)"""
        
        @staticmethod
        def is_displayed(player):
            return player.round_number == C.NUM_ROUNDS
        
        @staticmethod
        def vars_for_template(player):
            total_payoff = sum([p.payoff for p in player.in_all_rounds()])
            return dict(
                total_payoff=total_payoff,
                total_rounds=C.NUM_ROUNDS,
                real_money=int(total_payoff * player.session.config['real_world_currency_per_point']),
            )
    
    # 페이지 순서
    page_sequence = [Introduction, Contribute, ResultsWaitPage, Results, Survey, FinalResults]

else:
    # oTree 미설치 시 구조 확인용
    print("[O7] oTree가 설치되지 않았습니다. 'pip install otree'로 설치하세요.")
    print("[O7] AIPartner 클래스는 독립적으로 사용 가능합니다.")
    
    # AI 파트너 데모
    if __name__ == "__main__":
        print("\n--- AI Partner Demo ---")
        for condition in ['selfish', 'prosocial', 'meta_ranking']:
            ai = AIPartner(condition)
            contribs = [ai.decide(r, 50, 0.5) for r in range(1, 21)]
            print(f"  {condition:>12s}: mean={np.mean(contribs):.1f}, "
                  f"std={np.std(contribs):.1f}, "
                  f"range=[{min(contribs)}, {max(contribs)}]")
