import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import glob
from collections import defaultdict

class BBoxEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("바운딩 박스 편집 도구 - 전체 어노테이션")
        self.root.geometry("1800x1100")
        
        # 데이터 초기화
        self.current_annotation_index = 0
        self.annotations = []
        self.existing_annotations = {}  # 기존 어노테이션들
        self.current_image = None
        self.current_photo = None
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_bbox = [0, 0, 0, 0]
        self.original_bbox = [0, 0, 0, 0]
        self.show_existing_annotations = tk.BooleanVar(value=True)
        
        # 폴더 구조 데이터
        self.folder_structure = {}
        self.current_folder = None
        self.current_drug_code = None
        
        # GUI 구성
        self.setup_gui()
        self.load_all_annotations()
        
    def setup_gui(self):
        """GUI 구성"""
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 좌측 컨트롤 패널
        control_frame = ttk.Frame(main_frame, width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 폴더 선택 프레임
        folder_frame = ttk.LabelFrame(control_frame, text="폴더 선택")
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 폴더 콤보박스
        ttk.Label(folder_frame, text="어노테이션 폴더:").pack(anchor=tk.W, padx=5, pady=2)
        self.folder_var = tk.StringVar()
        self.folder_combo = ttk.Combobox(folder_frame, textvariable=self.folder_var, state="readonly", width=35)
        self.folder_combo.pack(fill=tk.X, padx=5, pady=2)
        self.folder_combo.bind('<<ComboboxSelected>>', self.on_folder_select)
        
        # 약품 코드 콤보박스
        ttk.Label(folder_frame, text="약품 코드:").pack(anchor=tk.W, padx=5, pady=2)
        self.drug_var = tk.StringVar()
        self.drug_combo = ttk.Combobox(folder_frame, textvariable=self.drug_var, state="readonly", width=35)
        self.drug_combo.pack(fill=tk.X, padx=5, pady=2)
        self.drug_combo.bind('<<ComboboxSelected>>', self.on_drug_select)
        
        # 어노테이션 목록
        ttk.Label(control_frame, text="어노테이션 목록", font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        
        # 약품별 그룹화된 목록
        self.annotation_listbox = tk.Listbox(control_frame, height=8, width=50)
        self.annotation_listbox.pack(fill=tk.X, pady=(0, 10))
        self.annotation_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)
        
        # 현재 어노테이션 정보
        info_frame = ttk.LabelFrame(control_frame, text="현재 어노테이션 정보")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=6, width=45)
        self.info_text.pack(padx=5, pady=5)
        
        # 약품 모양 정보 프레임
        shape_frame = ttk.LabelFrame(control_frame, text="약품 모양 정보 (참고용)")
        shape_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.shape_text = tk.Text(shape_frame, height=4, width=45)
        self.shape_text.pack(padx=5, pady=5)
        
        # 기존 어노테이션 표시 옵션
        ttk.Checkbutton(control_frame, text="기존 어노테이션 표시", 
                       variable=self.show_existing_annotations, 
                       command=self.toggle_existing_annotations).pack(pady=(0, 10))
        
        # 바운딩 박스 좌표 입력
        bbox_frame = ttk.LabelFrame(control_frame, text="바운딩 박스 좌표")
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
        
        # 버튼들
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="좌표 적용", command=self.apply_coordinates).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="원본 복원", command=self.restore_original).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="저장", command=self.save_annotation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="다음", command=self.next_annotation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="이전", command=self.prev_annotation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="기존 어노테이션 참조", command=self.show_reference_info).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="전체 저장", command=self.save_all_annotations).pack(fill=tk.X, pady=2)
        
        # 상태 표시
        self.status_var = tk.StringVar()
        self.status_var.set("준비됨")
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue").pack()
        
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
        
    def load_all_annotations(self):
        """전체 어노테이션 폴더 로드"""
        
        TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
        IMAGES_PATH = 'data/raw_data/train_images'
        
        if not os.path.exists(TRAIN_ANNOTATIONS_PATH):
            messagebox.showerror("오류", f"어노테이션 폴더를 찾을 수 없습니다: {TRAIN_ANNOTATIONS_PATH}")
            return
        
        # 폴더 구조 분석
        for item in os.listdir(TRAIN_ANNOTATIONS_PATH):
            item_path = os.path.join(TRAIN_ANNOTATIONS_PATH, item)
            if os.path.isdir(item_path) and item.endswith('_json'):
                
                folder_name = item.replace('_json', '')
                self.folder_structure[folder_name] = {}
                
                # 각 약품 코드 폴더 분석
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith('K-'):
                        drug_code = subitem[2:]  # K- 제거
                        
                        json_files = glob.glob(os.path.join(subitem_path, "*.json"))
                        if json_files:
                            self.folder_structure[folder_name][drug_code] = {
                                'path': subitem_path,
                                'json_files': json_files,
                                'count': len(json_files)
                            }
        
        # 폴더 콤보박스 업데이트
        folder_names = list(self.folder_structure.keys())
        folder_names.sort()
        self.folder_combo['values'] = folder_names
        
        if folder_names:
            self.folder_combo.set(folder_names[0])
            self.on_folder_select()
        
        self.status_var.set(f"로드됨: {len(folder_names)}개 폴더")
    
    def on_folder_select(self, event=None):
        """폴더 선택 시"""
        selected_folder = self.folder_var.get()
        if not selected_folder:
            return
        
        self.current_folder = selected_folder
        
        # 약품 코드 콤보박스 업데이트
        drug_codes = list(self.folder_structure[selected_folder].keys())
        drug_codes.sort()
        self.drug_combo['values'] = drug_codes
        
        if drug_codes:
            self.drug_combo.set(drug_codes[0])
            self.on_drug_select()
        
        self.status_var.set(f"폴더 선택: {selected_folder}")
    
    def on_drug_select(self, event=None):
        """약품 코드 선택 시"""
        selected_drug = self.drug_var.get()
        if not selected_drug or not self.current_folder:
            return
        
        self.current_drug_code = selected_drug
        
        # 해당 약품의 어노테이션들 로드
        self.load_drug_annotations()
        
        self.status_var.set(f"약품 선택: K-{selected_drug}")
    
    def load_drug_annotations(self):
        """선택된 약품의 어노테이션들 로드"""
        
        if not self.current_folder or not self.current_drug_code:
            return
        
        IMAGES_PATH = 'data/raw_data/train_images'
        drug_info = self.folder_structure[self.current_folder][self.current_drug_code]
        
        self.annotations = []
        
        for json_file in drug_info['json_files']:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'images' in data and 'annotations' in data:
                    image_info = data['images'][0]
                    annotation = data['annotations'][0]
                    
                    image_name = image_info['file_name']
                    image_path = os.path.join(IMAGES_PATH, image_name)
                    
                    if os.path.exists(image_path):
                        self.annotations.append({
                            'folder_name': self.current_folder,
                            'drug_code': self.current_drug_code,
                            'json_file': json_file,
                            'image_path': image_path,
                            'image_info': image_info,
                            'annotation': annotation,
                            'bbox': annotation['bbox'].copy(),
                            'category_id': annotation['category_id']
                        })
                        
            except Exception as e:
                print(f"❌ 오류: {json_file} - {e}")
        
        # 어노테이션 목록 업데이트
        self.update_annotation_list()
        
        if self.annotations:
            self.current_annotation_index = 0
            self.load_current_annotation()
        else:
            self.canvas.delete("all")
            self.info_text.delete(1.0, tk.END)
            self.shape_text.delete(1.0, tk.END)
            self.status_var.set("어노테이션이 없습니다")
    
    def update_annotation_list(self):
        """어노테이션 목록 업데이트"""
        self.annotation_listbox.delete(0, tk.END)
        
        for i, anno in enumerate(self.annotations):
            image_name = os.path.basename(anno['image_path'])
            display_text = f"{i+1:2d}. {image_name}"
            self.annotation_listbox.insert(tk.END, display_text)
    
    def on_annotation_select(self, event):
        """어노테이션 선택 시"""
        selection = self.annotation_listbox.curselection()
        if selection:
            self.current_annotation_index = selection[0]
            self.load_current_annotation()
    
    def load_current_annotation(self):
        """현재 어노테이션 로드"""
        if not self.annotations or self.current_annotation_index >= len(self.annotations):
            return
        
        anno = self.annotations[self.current_annotation_index]
        
        # 이미지 로드
        image = cv2.imread(anno['image_path'])
        if image is None:
            messagebox.showerror("오류", f"이미지를 로드할 수 없습니다: {anno['image_path']}")
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
        self.current_bbox = anno['bbox'].copy()
        self.original_bbox = anno['bbox'].copy()
        self.draw_bbox()
        
        # 정보 업데이트
        self.update_info_display(anno)
        self.update_shape_display(anno)
        self.update_coordinate_display()
        
        # 상태 업데이트
        self.status_var.set(f"로드됨: {anno['folder_name']} - K-{anno['drug_code']} - {os.path.basename(anno['image_path'])}")
    
    def update_shape_display(self, anno):
        """약품 모양 정보 표시 업데이트"""
        self.shape_text.delete(1.0, tk.END)
        
        image_info = anno['image_info']
        
        shape_info = f"현재 약품 (K-{anno['drug_code']}) 모양 정보:\n"
        shape_info += f"• dl_custom_shape: {image_info.get('dl_custom_shape', 'N/A')}\n"
        shape_info += f"• chart: {image_info.get('chart', 'N/A')}\n"
        shape_info += f"• drug_shape: {image_info.get('drug_shape', 'N/A')}\n"
        
        # 같은 폴더의 다른 약품들 정보도 추가
        folder_name = anno['folder_name']
        if folder_name in self.folder_structure:
            shape_info += f"\n같은 폴더의 다른 약품들:\n"
            for drug_code, info in self.folder_structure[folder_name].items():
                if drug_code != anno['drug_code']:
                    shape_info += f"• K-{drug_code}: {info['count']}개 어노테이션\n"
        
        self.shape_text.insert(1.0, shape_info)
    
    def draw_bbox(self):
        """바운딩 박스 그리기"""
        self.canvas.delete("bbox")
        self.canvas.delete("existing_bbox")
        
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            scaled_x = x * self.scale_factor
            scaled_y = y * self.scale_factor
            scaled_w = w * self.scale_factor
            scaled_h = h * self.scale_factor
            
            # 현재 바운딩 박스 그리기 (빨간색)
            self.canvas.create_rectangle(
                scaled_x, scaled_y, 
                scaled_x + scaled_w, scaled_y + scaled_h,
                outline="red", width=3, tags="bbox"
            )
            
            # 약품 코드 텍스트
            anno = self.annotations[self.current_annotation_index]
            self.canvas.create_text(
                scaled_x, scaled_y - 10,
                text=f"K-{anno['drug_code']} (현재)",
                fill="red", font=("Arial", 12, "bold"),
                anchor=tk.SW, tags="bbox"
            )
        
        # 기존 어노테이션들 표시 (같은 폴더의 다른 약품들)
        if self.show_existing_annotations.get() and self.current_folder:
            colors = ["blue", "green", "orange", "purple", "brown"]
            color_idx = 0
            
            for drug_code, info in self.folder_structure[self.current_folder].items():
                if drug_code != self.current_drug_code:  # 현재 편집 중인 약품 제외
                    color = colors[color_idx % len(colors)]
                    
                    # 해당 약품의 첫 번째 어노테이션에서 바운딩 박스 정보 가져오기
                    if info['json_files']:
                        try:
                            with open(info['json_files'][0], 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if 'annotations' in data:
                                ex_x, ex_y, ex_w, ex_h = data['annotations'][0]['bbox']
                                scaled_ex_x = ex_x * self.scale_factor
                                scaled_ex_y = ex_y * self.scale_factor
                                scaled_ex_w = ex_w * self.scale_factor
                                scaled_ex_h = ex_h * self.scale_factor
                                
                                # 기존 바운딩 박스 그리기 (점선)
                                self.canvas.create_rectangle(
                                    scaled_ex_x, scaled_ex_y,
                                    scaled_ex_x + scaled_ex_w, scaled_ex_y + scaled_ex_h,
                                    outline=color, width=2, dash=(5, 5), tags="existing_bbox"
                                )
                                
                                # 약품 코드 텍스트
                                self.canvas.create_text(
                                    scaled_ex_x, scaled_ex_y - 5,
                                    text=f"K-{drug_code}",
                                    fill=color, font=("Arial", 10),
                                    anchor=tk.SW, tags="existing_bbox"
                                )
                        except:
                            pass
                    
                    color_idx += 1
    
    def update_info_display(self, anno):
        """정보 표시 업데이트"""
        self.info_text.delete(1.0, tk.END)
        
        info = f"폴더: {anno['folder_name']}\n"
        info += f"약품 코드: K-{anno['drug_code']}\n"
        info += f"이미지: {os.path.basename(anno['image_path'])}\n"
        info += f"JSON 파일: {os.path.basename(anno['json_file'])}\n"
        info += f"카테고리 ID: {anno['category_id']}\n"
        info += f"현재 BBox: {anno['bbox']}\n\n"
        
        # 폴더 내 약품 정보 추가
        folder_name = anno['folder_name']
        if folder_name in self.folder_structure:
            info += "폴더 내 약품들:\n"
            for drug_code, info_data in self.folder_structure[folder_name].items():
                info += f"  K-{drug_code}: {info_data['count']}개\n"
        
        self.info_text.insert(1.0, info)
    
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
        if not self.annotations or self.current_annotation_index >= len(self.annotations):
            return
        
        anno = self.annotations[self.current_annotation_index]
        
        try:
            # JSON 파일 읽기
            with open(anno['json_file'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 바운딩 박스 업데이트
            data['annotations'][0]['bbox'] = self.current_bbox
            data['annotations'][0]['area'] = self.current_bbox[2] * self.current_bbox[3]
            
            # 파일 저장
            with open(anno['json_file'], 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            # 메모리 업데이트
            anno['bbox'] = self.current_bbox.copy()
            
            self.status_var.set("저장됨")
            messagebox.showinfo("성공", "어노테이션이 저장되었습니다.")
            
        except Exception as e:
            messagebox.showerror("오류", f"저장 중 오류가 발생했습니다: {e}")
    
    def save_all_annotations(self):
        """모든 어노테이션 저장"""
        if not self.annotations:
            messagebox.showinfo("정보", "저장할 어노테이션이 없습니다.")
            return
        
        saved_count = 0
        error_count = 0
        
        for anno in self.annotations:
            try:
                # JSON 파일 읽기
                with open(anno['json_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 바운딩 박스 업데이트
                data['annotations'][0]['bbox'] = anno['bbox']
                data['annotations'][0]['area'] = anno['bbox'][2] * anno['bbox'][3]
                
                # 파일 저장
                with open(anno['json_file'], 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                saved_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"저장 오류: {anno['json_file']} - {e}")
        
        message = f"저장 완료!\n성공: {saved_count}개\n실패: {error_count}개"
        messagebox.showinfo("전체 저장 완료", message)
        self.status_var.set(f"전체 저장 완료: {saved_count}개")
    
    def show_reference_info(self):
        """기존 어노테이션 참조 정보 표시"""
        if not self.current_folder:
            messagebox.showinfo("정보", "폴더를 먼저 선택해주세요.")
            return
        
        info = f"=== {self.current_folder} 폴더의 모든 약품 정보 ===\n\n"
        
        for drug_code, info_data in self.folder_structure[self.current_folder].items():
            info += f"K-{drug_code} ({info_data['count']}개 어노테이션):\n"
            
            # 첫 번째 어노테이션에서 모양 정보 가져오기
            if info_data['json_files']:
                try:
                    with open(info_data['json_files'][0], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'images' in data:
                        img_info = data['images'][0]
                        shape_info = f"  모양: {img_info.get('dl_custom_shape', 'N/A')} | "
                        shape_info += f"차트: {img_info.get('chart', 'N/A')} | "
                        shape_info += f"약품형태: {img_info.get('drug_shape', 'N/A')}\n"
                        info += shape_info
                except:
                    pass
            
            info += "\n"
        
        # 새 창으로 표시
        ref_window = tk.Toplevel(self.root)
        ref_window.title(f"폴더 정보 - {self.current_folder}")
        ref_window.geometry("800x600")
        
        text_widget = tk.Text(ref_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, info)
        
        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(ref_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
    
    def toggle_existing_annotations(self):
        """기존 어노테이션 표시 토글"""
        self.draw_bbox()
        
    def next_annotation(self):
        """다음 어노테이션"""
        if self.current_annotation_index < len(self.annotations) - 1:
            self.current_annotation_index += 1
            self.load_current_annotation()
            self.annotation_listbox.selection_clear(0, tk.END)
            self.annotation_listbox.selection_set(self.current_annotation_index)
    
    def prev_annotation(self):
        """이전 어노테이션"""
        if self.current_annotation_index > 0:
            self.current_annotation_index -= 1
            self.load_current_annotation()
            self.annotation_listbox.selection_clear(0, tk.END)
            self.annotation_listbox.selection_set(self.current_annotation_index)

def main():
    """메인 실행 함수"""
    root = tk.Tk()
    app = BBoxEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main() 