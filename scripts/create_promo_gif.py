import os
import glob
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

def create_gif(output_path, duration=1.5):
    base_dir = "simulation/outputs"
    # 숫자로 시작하는 run_large 폴더만 선택 (harvest 제외)
    runs = glob.glob(os.path.join(base_dir, "run_large_1*"))
    
    # 생성 시간순 정렬
    runs.sort(key=os.path.getmtime)
    
    if len(runs) < 2:
        print("Not enough runs found.")
        return

    # 예상: [-2] = ON (11:12), [-1] = OFF (11:55)
    # 하지만 사용자가 실행한 순서는 ON -> OFF 이므로 시간순 정렬이 맞음
    run_on = runs[-2]
    run_off = runs[-1]
    
    print(f"ON Run: {run_on}")
    print(f"OFF Run: {run_off}")
    
    fig_name = "fig2_cooperation_rate.png"
    
    path_on = os.path.join(run_on, "figures", fig_name)
    path_off = os.path.join(run_off, "figures", fig_name)
    
    # Fallback: ON 폴더에 그림이 없으면 reproduce에서 찾기
    if not os.path.exists(path_on):
        print(f"Warning: {path_on} not found. Trying reproduce folder.")
        path_on = os.path.join(base_dir, "reproduce", "figures", fig_name)

    if not os.path.exists(path_on) or not os.path.exists(path_off):
        print(f"Error: Figures missing.\nON: {path_on}\nOFF: {path_off}")
        return

    # 이미지 로드
    img_on = Image.open(path_on).convert("RGB")
    img_off = Image.open(path_off).convert("RGB")
    
    # 텍스트 추가 함수
    def add_text(img, text, color=(0, 128, 0)):
        draw = ImageDraw.Draw(img)
        try:
            # 폰트 크기 키움
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        # 텍스트 위치
        draw.text((50, 50), text, fill=color, font=font, stroke_width=3, stroke_fill="white")
        return img

    # 텍스트 추가
    img_on = add_text(img_on, "Phase 1: EthicaAI (Meta-Ranking ON)", color=(0, 0, 255))
    img_off = add_text(img_off, "Phase 2: Baseline (Meta-Ranking OFF)", color=(255, 0, 0))

    # 프레임 구성
    frames = [img_on, img_off]
    
    # GIF 저장
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
    print(f"Successfully saved GIF to {output_path}")

if __name__ == "__main__":
    # 저장 경로를 reproduce/figures로 설정하여 대시보드에서도 쓸 수 있게 함
    output_dir = "simulation/outputs/reproduce/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_gif = os.path.join(output_dir, "promo_cooperation.gif")
    create_gif(output_gif)
