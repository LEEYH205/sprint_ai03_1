import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import glob
from collections import defaultdict

class DrugCodeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("약품코드별 이미지 뷰어")
        self.root.geometry("1600x1000")
        
        # 데이터 초기화
        self.drug_codes = []
        self.current_drug_code = None
        self.current_image_index = 0
        self.images_data = []
        self.current_image = None
        self.current_photo = None
        self.scale_factor = 1.0
        
        # 바운딩 박스 편집 관련
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_bbox = [0, 0, 0, 0]
        self.original_bbox = [0, 0, 0, 0]
        
        # GUI 구성
        self.setup_gui()
        self.load_drug_codes()
        
    def setup_gui(self):
        """GUI 구성"""
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 컨트롤 패널
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 약품코드 선택 프레임
        drug_frame = ttk.LabelFrame(control_frame, text="약품코드 선택")
        drug_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Label(drug_frame, text="약품코드:").pack(side=tk.LEFT, padx=5)
        self.drug_var = tk.StringVar()
        self.drug_combo = ttk.Combobox(drug_frame, textvariable=self.drug_var, state="readonly", width=20)
        self.drug_combo.pack(side=tk.LEFT, padx=5)
        self.drug_combo.bind('<<ComboboxSelected>>', self.on_drug_select)
        
        # 이미지 네비게이션 프레임
        nav_frame = ttk.LabelFrame(control_frame, text="이미지 네비게이션")
        nav_frame.pack(side=tk.LEFT, fill=tk.X, padx=(0, 10))
        
        ttk.Button(nav_frame, text="◀ 이전", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        self.image_info_label = ttk.Label(nav_frame, text="0 / 0")
        self.image_info_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="다음 ▶", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # 편집 버튼 프레임
        edit_frame = ttk.LabelFrame(control_frame, text="편집")
        edit_frame.pack(side=tk.LEFT, fill=tk.X, padx=(0, 10))
        
        ttk.Button(edit_frame, text="원본 복원", command=self.restore_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_frame, text="저장", command=self.save_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_frame, text="BBox 모아보기", command=self.show_bbox_collection).pack(side=tk.LEFT, padx=5)
        
        # 좌측 정보 패널
        info_frame = ttk.Frame(main_frame, width=300)
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 현재 이미지 정보
        current_info_frame = ttk.LabelFrame(info_frame, text="현재 이미지 정보")
        current_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_info_text = tk.Text(current_info_frame, height=8, width=35)
        self.current_info_text.pack(padx=5, pady=5)
        
        # 바운딩 박스 좌표
        bbox_frame = ttk.LabelFrame(info_frame, text="바운딩 박스 좌표")
        bbox_frame.pack(fill=tk.X, pady=(0, 10))
        
        # X 좌표
        ttk.Label(bbox_frame, text="X:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.x_var = tk.StringVar()
        self.x_entry = ttk.Entry(bbox_frame, textvariable=self.x_var, width=10)
        self.x_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Y 좌표
        ttk.Label(bbox_frame, text="Y:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.y_var = tk.StringVar()
        self.y_entry = ttk.Entry(bbox_frame, textvariable=self.y_var, width=10)
        self.y_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Width
        ttk.Label(bbox_frame, text="Width:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.width_var = tk.StringVar()
        self.width_entry = ttk.Entry(bbox_frame, textvariable=self.width_var, width=10)
        self.width_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Height
        ttk.Label(bbox_frame, text="Height:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.height_var = tk.StringVar()
        self.height_entry = ttk.Entry(bbox_frame, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # 좌표 적용 버튼
        ttk.Button(bbox_frame, text="좌표 적용", command=self.apply_coordinates).grid(row=4, column=0, columnspan=2, pady=5)
        
        # 이미지 목록
        list_frame = ttk.LabelFrame(info_frame, text="이미지 목록")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_listbox = tk.Listbox(list_frame, width=35)
        self.image_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 스크롤바
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        # 우측 이미지 패널
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 이미지 캔버스
        self.canvas = tk.Canvas(image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 마우스 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 스크롤바
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 상태 표시
        self.status_var = tk.StringVar()
        self.status_var.set("준비됨")
        ttk.Label(main_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.BOTTOM, pady=5)
        
    def load_drug_codes(self):
        """약품코드 목록 로드"""
        
        TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
        
        if not os.path.exists(TRAIN_ANNOTATIONS_PATH):
            messagebox.showerror("오류", f"어노테이션 폴더를 찾을 수 없습니다: {TRAIN_ANNOTATIONS_PATH}")
            return
        
        # 모든 약품코드 수집
        drug_codes_set = set()
        
        for item in os.listdir(TRAIN_ANNOTATIONS_PATH):
            item_path = os.path.join(TRAIN_ANNOTATIONS_PATH, item)
            if os.path.isdir(item_path) and item.endswith('_json'):
                
                # 각 약품 코드 폴더 분석
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith('K-'):
                        drug_code = subitem[2:]  # K- 제거
                        drug_codes_set.add(drug_code)
        
        # 약품코드 정렬
        self.drug_codes = sorted(list(drug_codes_set))
        
        # 드롭다운 업데이트
        self.drug_combo['values'] = self.drug_codes
        
        if self.drug_codes:
            self.drug_combo.set(self.drug_codes[0])
            self.on_drug_select()
        
        self.status_var.set(f"로드됨: {len(self.drug_codes)}개 약품코드")
    
    def on_drug_select(self, event=None):
        """약품코드 선택 시"""
        selected_drug = self.drug_var.get()
        if not selected_drug:
            return
        
        self.current_drug_code = selected_drug
        self.load_drug_images()
        self.status_var.set(f"약품코드 선택: K-{selected_drug}")
    
    def load_drug_images(self):
        """선택된 약품코드의 모든 이미지 로드"""
        
        if not self.current_drug_code:
            return
        
        TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
        IMAGES_PATH = 'data/raw_data/train_images'
        
        self.images_data = []
        
        # 모든 어노테이션 폴더에서 해당 약품코드 찾기
        for item in os.listdir(TRAIN_ANNOTATIONS_PATH):
            item_path = os.path.join(TRAIN_ANNOTATIONS_PATH, item)
            if os.path.isdir(item_path) and item.endswith('_json'):
                
                drug_folder = os.path.join(item_path, f"K-{self.current_drug_code}")
                if os.path.exists(drug_folder):
                    
                    # 해당 약품의 모든 JSON 파일 처리
                    json_files = glob.glob(os.path.join(drug_folder, "*.json"))
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if 'images' in data and 'annotations' in data:
                                image_info = data['images'][0]
                                annotation = data['annotations'][0]
                                
                                image_name = image_info['file_name']
                                image_path = os.path.join(IMAGES_PATH, image_name)
                                
                                if os.path.exists(image_path):
                                    self.images_data.append({
                                        'folder_name': item.replace('_json', ''),
                                        'json_file': json_file,
                                        'image_path': image_path,
                                        'image_name': image_name,
                                        'image_info': image_info,
                                        'annotation': annotation,
                                        'bbox': annotation['bbox'].copy(),
                                        'category_id': annotation['category_id']
                                    })
                                    
                        except Exception as e:
                            print(f"❌ 오류: {json_file} - {e}")
        
        # 이미지 목록 업데이트
        self.update_image_list()
        
        if self.images_data:
            self.current_image_index = 0
            self.load_current_image()
        else:
            self.canvas.delete("all")
            self.current_info_text.delete(1.0, tk.END)
            self.status_var.set("해당 약품코드의 이미지가 없습니다")
    
    def update_image_list(self):
        """이미지 목록 업데이트"""
        self.image_listbox.delete(0, tk.END)
        
        for i, img_data in enumerate(self.images_data):
            display_text = f"{i+1:2d}. {img_data['image_name']}"
            self.image_listbox.insert(tk.END, display_text)
        
        # 이미지 정보 업데이트
        self.image_info_label.config(text=f"{self.current_image_index + 1} / {len(self.images_data)}")
    
    def on_image_select(self, event):
        """이미지 선택 시"""
        selection = self.image_listbox.curselection()
        if selection:
            self.current_image_index = selection[0]
            self.load_current_image()
    
    def load_current_image(self):
        """현재 이미지 로드"""
        if not self.images_data or self.current_image_index >= len(self.images_data):
            return
        
        img_data = self.images_data[self.current_image_index]
        
        # 이미지 로드
        image = cv2.imread(img_data['image_path'])
        if image is None:
            messagebox.showerror("오류", f"이미지를 로드할 수 없습니다: {img_data['image_path']}")
            return
        
        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image_rgb)
        
        # 이미지 크기 조정 (캔버스에 맞게)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # 캔버스 크기에 맞게 이미지 조정
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.scale_factor = scale
        else:
            self.scale_factor = 1.0
        
        # PhotoImage로 변환
        self.current_photo = ImageTk.PhotoImage(pil_image)
        
        # 캔버스에 이미지 표시
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        
        # 바운딩 박스 그리기
        self.current_bbox = img_data['bbox'].copy()
        self.original_bbox = img_data['bbox'].copy()
        self.draw_bbox()
        
        # 정보 업데이트
        self.update_info_display(img_data)
        self.update_coordinate_display()
        
        # 목록 선택 업데이트
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_image_index)
        
        # 이미지 정보 업데이트
        self.image_info_label.config(text=f"{self.current_image_index + 1} / {len(self.images_data)}")
        
        # 상태 업데이트
        self.status_var.set(f"로드됨: K-{self.current_drug_code} - {img_data['image_name']}")
    
    def draw_bbox(self):
        """바운딩 박스 그리기"""
        self.canvas.delete("bbox")
        
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            scaled_x = x * self.scale_factor
            scaled_y = y * self.scale_factor
            scaled_w = w * self.scale_factor
            scaled_h = h * self.scale_factor
            
            # 바운딩 박스 그리기 (빨간색)
            self.canvas.create_rectangle(
                scaled_x, scaled_y, 
                scaled_x + scaled_w, scaled_y + scaled_h,
                outline="red", width=3, tags="bbox"
            )
            
            # 약품코드 텍스트
            self.canvas.create_text(
                scaled_x, scaled_y - 10,
                text=f"K-{self.current_drug_code}",
                fill="red", font=("Arial", 12, "bold"),
                anchor=tk.SW, tags="bbox"
            )
    
    def update_info_display(self, img_data):
        """정보 표시 업데이트"""
        self.current_info_text.delete(1.0, tk.END)
        
        info = f"약품코드: K-{self.current_drug_code}\n"
        info += f"폴더: {img_data['folder_name']}\n"
        info += f"이미지: {img_data['image_name']}\n"
        info += f"카테고리 ID: {img_data['category_id']}\n"
        info += f"바운딩 박스: {img_data['bbox']}\n\n"
        
        # 이미지 정보 추가
        image_info = img_data['image_info']
        info += "이미지 정보:\n"
        info += f"• 모양: {image_info.get('dl_custom_shape', 'N/A')}\n"
        info += f"• 차트: {image_info.get('chart', 'N/A')}\n"
        info += f"• 약품형태: {image_info.get('drug_shape', 'N/A')}\n"
        info += f"• 크기: {image_info.get('width', 'N/A')} x {image_info.get('height', 'N/A')}\n"
        
        self.current_info_text.insert(1.0, info)
    
    def update_coordinate_display(self):
        """좌표 표시 업데이트"""
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            self.x_var.set(f"{x:.1f}")
            self.y_var.set(f"{y:.1f}")
            self.width_var.set(f"{w:.1f}")
            self.height_var.set(f"{h:.1f}")
    
    def on_mouse_down(self, event):
        """마우스 다운"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
    
    def on_mouse_drag(self, event):
        """마우스 드래그"""
        if self.drawing:
            # 임시 바운딩 박스 그리기
            self.canvas.delete("temp_bbox")
            self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline="blue", width=2, tags="temp_bbox"
            )
    
    def on_mouse_up(self, event):
        """마우스 업"""
        if self.drawing:
            self.drawing = False
            
            # 임시 바운딩 박스 제거
            self.canvas.delete("temp_bbox")
            
            # 새로운 바운딩 박스 계산
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            # 스케일 팩터로 원본 좌표로 변환
            x = x1 / self.scale_factor
            y = y1 / self.scale_factor
            w = (x2 - x1) / self.scale_factor
            h = (y2 - y1) / self.scale_factor
            
            # 바운딩 박스 업데이트
            self.current_bbox = [x, y, w, h]
            self.draw_bbox()
            self.update_coordinate_display()
            
            self.status_var.set("바운딩 박스 업데이트됨")
    
    def apply_coordinates(self):
        """좌표 적용"""
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            w = float(self.width_var.get())
            h = float(self.height_var.get())
            
            self.current_bbox = [x, y, w, h]
            self.draw_bbox()
            self.status_var.set("좌표 적용됨")
            
        except ValueError:
            messagebox.showerror("오류", "올바른 숫자를 입력하세요.")
    
    def restore_original(self):
        """원본 복원"""
        self.current_bbox = self.original_bbox.copy()
        self.draw_bbox()
        self.update_coordinate_display()
        self.status_var.set("원본 복원됨")
    
    def save_annotation(self):
        """어노테이션 저장"""
        if not self.images_data or self.current_image_index >= len(self.images_data):
            return
        
        img_data = self.images_data[self.current_image_index]
        
        try:
            # JSON 파일 읽기
            with open(img_data['json_file'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 바운딩 박스 업데이트
            data['annotations'][0]['bbox'] = self.current_bbox
            data['annotations'][0]['area'] = self.current_bbox[2] * self.current_bbox[3]
            
            # 파일 저장
            with open(img_data['json_file'], 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            # 메모리 업데이트
            img_data['bbox'] = self.current_bbox.copy()
            
            self.status_var.set("저장됨")
            messagebox.showinfo("성공", "어노테이션이 저장되었습니다.")
            
        except Exception as e:
            messagebox.showerror("오류", f"저장 중 오류가 발생했습니다: {e}")
    
    def next_image(self):
        """다음 이미지"""
        if self.current_image_index < len(self.images_data) - 1:
            self.current_image_index += 1
            self.load_current_image()
    
    def prev_image(self):
        """이전 이미지"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def show_bbox_collection(self):
        """바운딩 박스 모아보기 창"""
        if not self.images_data:
            messagebox.showinfo("정보", "표시할 이미지가 없습니다.")
            return
        
        # 새 창 생성
        bbox_window = tk.Toplevel(self.root)
        bbox_window.title(f"바운딩 박스 모아보기 - K-{self.current_drug_code}")
        bbox_window.geometry("1400x900")
        
        # 메인 프레임
        main_frame = ttk.Frame(bbox_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 정보
        info_label = ttk.Label(main_frame, text=f"K-{self.current_drug_code} - 총 {len(self.images_data)}개 이미지", 
                              font=('Arial', 14, 'bold'))
        info_label.pack(pady=(0, 10))
        
        # 캔버스와 스크롤바를 담을 프레임
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 캔버스 생성
        canvas = tk.Canvas(canvas_frame, bg="white")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 스크롤바
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 내부 프레임 (스크롤 가능한 영역)
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
        
        # 이미지들을 그리드로 배치
        cols = 4  # 한 행에 4개씩
        current_row = 0
        current_col = 0
        max_width = 0
        total_height = 0
        
        for i, img_data in enumerate(self.images_data):
            try:
                # 이미지 로드
                image = cv2.imread(img_data['image_path'])
                if image is None:
                    continue
                
                # BGR to RGB 변환
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 바운딩 박스 좌표
                bbox = img_data['bbox']
                x, y, w, h = bbox
                
                # 바운딩 박스 영역만 크롭
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # 이미지 경계 확인
                img_height, img_width = image_rgb.shape[:2]
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                # 크롭된 이미지
                cropped_image = image_rgb[y1:y2, x1:x2]
                
                if cropped_image.size == 0:
                    # 크롭 실패시 전체 이미지 사용
                    cropped_image = image_rgb
                
                # PIL Image로 변환
                pil_image = Image.fromarray(cropped_image)
                
                # 크기 조정 (최대 200x200)
                max_size = 200
                img_width, img_height = pil_image.size
                if img_width > max_size or img_height > max_size:
                    scale = min(max_size / img_width, max_size / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # PhotoImage로 변환
                photo = ImageTk.PhotoImage(pil_image)
                
                # 이미지 프레임 생성
                img_frame = ttk.Frame(inner_frame)
                img_frame.grid(row=current_row, column=current_col, padx=5, pady=5, sticky="nsew")
                
                # 이미지 라벨
                img_label = ttk.Label(img_frame, image=photo)
                img_label.image = photo  # 참조 유지
                img_label.pack()
                
                # 정보 라벨
                info_text = f"{i+1:2d}. {img_data['image_name']}\n"
                info_text += f"BBox: [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]"
                info_label = ttk.Label(img_frame, text=info_text, font=('Arial', 8))
                info_label.pack()
                
                # 더블클릭 이벤트 (원본 창에서 해당 이미지로 이동)
                img_label.bind("<Double-Button-1>", lambda e, idx=i: self.jump_to_image(idx, bbox_window))
                
                # 다음 위치 계산
                current_col += 1
                if current_col >= cols:
                    current_col = 0
                    current_row += 1
                
                # 최대 크기 업데이트
                frame_width = img_frame.winfo_reqwidth()
                frame_height = img_frame.winfo_reqheight()
                max_width = max(max_width, frame_width)
                total_height = (current_row + 1) * (frame_height + 10)
                
            except Exception as e:
                print(f"이미지 처리 오류: {img_data['image_path']} - {e}")
                continue
        
        # 스크롤 영역 설정
        canvas.configure(scrollregion=(0, 0, max_width * cols, total_height))
        
        # 마우스 휠 스크롤
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        # 창 크기 조정 시 캔버스 업데이트
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        inner_frame.bind("<Configure>", on_configure)
    
    def jump_to_image(self, image_index, bbox_window):
        """BBox 모아보기에서 원본 창으로 해당 이미지로 이동"""
        self.current_image_index = image_index
        self.load_current_image()
        bbox_window.destroy()
        self.root.focus_force()

def main():
    """메인 실행 함수"""
    root = tk.Tk()
    app = DrugCodeViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 