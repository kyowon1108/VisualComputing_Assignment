"""
테스트 이미지 다운로드 스크립트
Test Image Download Script

이 스크립트는 히스토그램 평활화와 Otsu Thresholding 테스트를 위한
다양한 샘플 이미지를 Unsplash에서 자동으로 다운로드합니다.

This script automatically downloads various sample images from Unsplash
for testing histogram equalization and Otsu thresholding.
"""

import os
import requests
import shutil
from urllib.parse import urlparse
import time
from typing import List, Dict

# 테스트 이미지 정보 / Test image information
TEST_IMAGES = [
    {
        'name': 'low_contrast_landscape.jpg',
        'url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80',
        'description': '저대비 풍경 이미지 (히스토그램 평활화 테스트용) / Low contrast landscape (for HE testing)'
    },
    {
        'name': 'high_contrast_portrait.jpg',
        'url': 'https://images.unsplash.com/photo-1544717297-fa95b6ee9643?w=800&q=80',
        'description': '고대비 인물 이미지 (CLAHE 테스트용) / High contrast portrait (for CLAHE testing)'
    },
    {
        'name': 'mixed_lighting_architecture.jpg',
        'url': 'https://images.unsplash.com/photo-1551632811-561732d1e306?w=800&q=80',
        'description': '혼합 조명 건축물 (Local Otsu 테스트용) / Mixed lighting architecture (for Local Otsu testing)'
    },
    {
        'name': 'text_document.jpg',
        'url': 'https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=800&q=80',
        'description': '텍스트 문서 이미지 (이진화 테스트용) / Text document (for binarization testing)'
    },
    {
        'name': 'nature_macro.jpg',
        'url': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&q=80',
        'description': '자연 매크로 이미지 (컬러 HE 테스트용) / Nature macro (for color HE testing)'
    },
    {
        'name': 'urban_night.jpg',
        'url': 'https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800&q=80',
        'description': '도시 야경 (저조도 이미지 테스트용) / Urban night scene (for low-light testing)'
    }
]

def create_directories():
    """
    필요한 디렉토리 생성
    Create necessary directories
    """
    directories = ['images', 'results', 'tests']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ 디렉토리 생성: {directory} / Directory created: {directory}")

def download_image(image_info: Dict[str, str], timeout: int = 30) -> bool:
    """
    단일 이미지 다운로드
    Download a single image

    Args:
        image_info (Dict[str, str]): 이미지 정보 (name, url, description)
        timeout (int): 타임아웃 (초)

    Returns:
        bool: 다운로드 성공 여부
    """
    try:
        image_path = os.path.join('images', image_info['name'])

        # 이미 존재하는 경우 스킵 / Skip if already exists
        if os.path.exists(image_path):
            print(f"⚠️  이미지가 이미 존재합니다: {image_info['name']} / Image already exists")
            return True

        print(f"📥 다운로드 중: {image_info['name']} / Downloading...")
        print(f"   URL: {image_info['url']}")
        print(f"   설명: {image_info['description']}")

        # User-Agent 헤더 추가 / Add User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # 이미지 다운로드 / Download image
        response = requests.get(image_info['url'], headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        # 파일로 저장 / Save to file
        with open(image_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

        # 파일 크기 확인 / Check file size
        file_size = os.path.getsize(image_path) / 1024  # KB
        print(f"✓ 다운로드 완료: {image_info['name']} ({file_size:.1f} KB)")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ 네트워크 오류: {image_info['name']} - {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 다운로드 실패: {image_info['name']} - {str(e)}")
        return False

def create_sample_images():
    """
    프로그래밍 방식으로 샘플 이미지 생성
    Create sample images programmatically
    """
    try:
        import numpy as np
        from PIL import Image

        print("\n🎨 프로그래밍 방식 샘플 이미지 생성 중... / Creating programmatic sample images...")

        # 1. 저대비 그라디언트 이미지 / Low contrast gradient image
        print("   📷 저대비 그라디언트 이미지 생성 / Creating low contrast gradient")
        gradient = np.linspace(80, 180, 400, dtype=np.uint8)
        gradient_2d = np.tile(gradient, (300, 1))
        gradient_rgb = np.stack([gradient_2d] * 3, axis=-1)
        Image.fromarray(gradient_rgb).save('images/generated_low_contrast.png')

        # 2. 체스판 패턴 (이진화 테스트용) / Checkerboard pattern (for binarization test)
        print("   📷 체스판 패턴 이미지 생성 / Creating checkerboard pattern")
        checkerboard = np.zeros((400, 400), dtype=np.uint8)
        check_size = 25
        for i in range(0, 400, check_size):
            for j in range(0, 400, check_size):
                if (i // check_size + j // check_size) % 2 == 0:
                    checkerboard[i:i+check_size, j:j+check_size] = 255
        # 노이즈 추가 / Add noise
        noise = np.random.normal(0, 10, checkerboard.shape)
        checkerboard_noisy = np.clip(checkerboard + noise, 0, 255).astype(np.uint8)
        checkerboard_rgb = np.stack([checkerboard_noisy] * 3, axis=-1)
        Image.fromarray(checkerboard_rgb).save('images/generated_checkerboard.png')

        # 3. 가우시안 노이즈 이미지 / Gaussian noise image
        print("   📷 노이즈 이미지 생성 / Creating noise image")
        noise_image = np.random.normal(128, 30, (300, 400, 3))
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
        Image.fromarray(noise_image).save('images/generated_noise.png')

        # 4. 혼합 조명 시뮬레이션 / Mixed lighting simulation
        print("   📷 혼합 조명 이미지 생성 / Creating mixed lighting image")
        x, y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 300))
        lighting = 100 + 80 * np.sin(x) * np.cos(y) + 50 * np.exp(-(x**2 + y**2))
        lighting = np.clip(lighting, 50, 200).astype(np.uint8)
        lighting_rgb = np.stack([lighting, lighting * 0.9, lighting * 0.8], axis=-1).astype(np.uint8)
        Image.fromarray(lighting_rgb).save('images/generated_mixed_lighting.png')

        print("✓ 프로그래밍 방식 샘플 이미지 생성 완료 / Programmatic sample images created")

    except ImportError:
        print("⚠️  PIL 또는 numpy가 설치되지 않아 샘플 이미지를 생성할 수 없습니다.")
        print("   pip install Pillow numpy를 실행하세요.")
    except Exception as e:
        print(f"❌ 샘플 이미지 생성 실패: {str(e)}")

def create_readme_for_images():
    """
    이미지 폴더에 README 파일 생성
    Create README file for images folder
    """
    readme_content = """# 테스트 이미지 / Test Images

이 폴더에는 히스토그램 평활화와 Local Otsu Thresholding 테스트를 위한 이미지들이 포함되어 있습니다.
This folder contains images for testing histogram equalization and Local Otsu thresholding.

## 다운로드된 이미지 / Downloaded Images

### 히스토그램 평활화 테스트용 / For Histogram Equalization Testing
- `low_contrast_landscape.jpg`: 저대비 풍경 이미지
- `high_contrast_portrait.jpg`: 고대비 인물 이미지 (CLAHE 효과 확인용)
- `nature_macro.jpg`: 컬러 히스토그램 평활화 테스트용

### Local Otsu Thresholding 테스트용 / For Local Otsu Thresholding Testing
- `mixed_lighting_architecture.jpg`: 혼합 조명 건축물
- `text_document.jpg`: 텍스트 문서 (이진화 성능 확인)
- `urban_night.jpg`: 저조도 도시 야경

### 생성된 샘플 이미지 / Generated Sample Images
- `generated_low_contrast.png`: 저대비 그라디언트
- `generated_checkerboard.png`: 체스판 패턴 (노이즈 포함)
- `generated_noise.png`: 가우시안 노이즈
- `generated_mixed_lighting.png`: 혼합 조명 시뮬레이션

## 사용법 / Usage

1. GUI 애플리케이션에서 "예시 이미지" 버튼 클릭
2. 또는 "이미지 로드" 버튼으로 원하는 이미지 선택
3. 각 이미지별로 최적의 파라미터 테스트 권장

## 이미지 특성 / Image Characteristics

### 히스토그램 평활화 테스트 시 권장 설정 / Recommended Settings for HE Testing
- 저대비 이미지: YUV 색공간, CLAHE 비활성화
- 고대비 이미지: YUV 색공간, CLAHE 활성화 (Clip Limit: 2-3)
- 컬러 이미지: YUV 색공간 (색감 보존 확인)

### Local Otsu 테스트 시 권장 설정 / Recommended Settings for Local Otsu Testing
- 텍스트 이미지: Block-based (32x32 블록)
- 혼합 조명: Sliding Window (스트라이드 8-16)
- 복잡한 장면: 모든 방법 비교 모드

## 추가 이미지 / Additional Images

더 많은 테스트 이미지가 필요하다면:
1. `download_images.py` 스크립트 재실행
2. 또는 직접 이미지 파일을 이 폴더에 추가

For more test images:
1. Re-run the `download_images.py` script
2. Or manually add image files to this folder
"""

    readme_path = os.path.join('images', 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ 이미지 폴더 README 생성 완료 / Images folder README created")

def verify_downloads():
    """
    다운로드된 이미지들 검증
    Verify downloaded images
    """
    print("\n🔍 다운로드된 이미지 검증 중... / Verifying downloaded images...")

    images_dir = 'images'
    if not os.path.exists(images_dir):
        print("❌ images 폴더가 존재하지 않습니다 / Images folder does not exist")
        return

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print("❌ 이미지 파일이 없습니다 / No image files found")
        return

    print(f"✓ 총 {len(image_files)}개의 이미지 파일 발견 / Found {len(image_files)} image files:")

    total_size = 0
    for img_file in sorted(image_files):
        img_path = os.path.join(images_dir, img_file)
        file_size = os.path.getsize(img_path) / 1024  # KB
        total_size += file_size

        # 이미지 유효성 간단 체크 / Simple image validity check
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
                print(f"   📷 {img_file}: {width}x{height} ({mode}) - {file_size:.1f} KB")
        except ImportError:
            print(f"   📷 {img_file}: {file_size:.1f} KB")
        except Exception as e:
            print(f"   ❌ {img_file}: 손상된 파일 / Corrupted file - {str(e)}")

    print(f"✓ 총 이미지 크기: {total_size:.1f} KB / Total image size: {total_size:.1f} KB")

def main():
    """
    메인 함수 - 테스트 이미지 다운로드 프로세스 실행
    Main function - Execute test image download process
    """
    print("🚀 비쥬얼컴퓨팅 과제1 테스트 이미지 다운로드 시작")
    print("   Visual Computing Assignment 1 Test Image Download Started")
    print("=" * 70)

    # 1. 디렉토리 생성 / Create directories
    create_directories()

    # 2. 온라인 이미지 다운로드 시도 / Try to download online images
    print("\n📡 온라인 이미지 다운로드 시도 중... / Attempting to download online images...")

    success_count = 0
    fail_count = 0

    for i, image_info in enumerate(TEST_IMAGES, 1):
        print(f"\n[{i}/{len(TEST_IMAGES)}] ", end="")

        if download_image(image_info):
            success_count += 1
        else:
            fail_count += 1

        # 요청 간 지연 (서버 부하 방지) / Delay between requests (prevent server overload)
        if i < len(TEST_IMAGES):
            time.sleep(1)

    # 3. 결과 요약 / Results summary
    print(f"\n📊 다운로드 결과 요약 / Download Results Summary:")
    print(f"   ✓ 성공: {success_count}개 / Success: {success_count} images")
    print(f"   ❌ 실패: {fail_count}개 / Failed: {fail_count} images")

    # 4. 프로그래밍 방식 샘플 이미지 생성 / Create programmatic sample images
    create_sample_images()

    # 5. 이미지 폴더 README 생성 / Create images folder README
    create_readme_for_images()

    # 6. 다운로드 검증 / Verify downloads
    verify_downloads()

    # 7. 완료 메시지 / Completion message
    print("\n" + "=" * 70)
    print("🎉 테스트 이미지 준비 완료! / Test images preparation completed!")
    print("\n📝 다음 단계 / Next steps:")
    print("   1. python main.py 실행하여 GUI 애플리케이션 시작")
    print("   2. '예시 이미지' 버튼으로 다운로드된 이미지 테스트")
    print("   3. 다양한 파라미터로 알고리즘 성능 비교")
    print("\n   1. Run 'python main.py' to start GUI application")
    print("   2. Test downloaded images using 'Example Image' buttons")
    print("   3. Compare algorithm performance with various parameters")

    # 8. 문제 해결 안내 / Troubleshooting guide
    if fail_count > 0:
        print(f"\n⚠️  일부 이미지 다운로드 실패 / Some images failed to download:")
        print("   - 인터넷 연결 확인 / Check internet connection")
        print("   - 방화벽 설정 확인 / Check firewall settings")
        print("   - 생성된 샘플 이미지로도 테스트 가능 / Can test with generated sample images")

if __name__ == "__main__":
    main()