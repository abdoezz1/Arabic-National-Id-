import easyocr
import cv2
import numpy as np
from typing import Dict, List, Tuple
from ultralytics import YOLO

class SimplifiedArabicIDDetector:
    def __init__(self, main_model_path: str, digits_model_path: str,enable_manual_correction: bool = True):
        """
        Initialize the simplified two-stage detector

        Args:
            main_model_path: Path to the main YOLO model (detects fields + ID card)
            digits_model_path: Path to the digits-only YOLO model
        """
        self.main_model = YOLO(main_model_path)
        self.digits_model = YOLO(digits_model_path)
        self.reader = easyocr.Reader(['ar'], gpu=True)
        self.enable_manual_correction = enable_manual_correction

        self.confidence_thresholds = {
            'national_id': 0.3,
            'first_name': 0.4,
            'last_name': 0.4,
            'address1': 0.3,
            'address2': 0.3,
            'id': 0.5,
            'digit': 0.3
        }

        self.digit_class_mapping = self._initialize_digit_mapping()
        self.expected_digits = 14

    def _initialize_digit_mapping(self) -> Dict[int, str]:
        """Initialize mapping from class IDs to digit values"""
        digit_mapping = {}
        for class_id, class_name in self.digits_model.names.items():
            if class_name.isdigit():
                digit_mapping[class_id] = class_name
        return digit_mapping

    def calculate_circular_std(self, angles):
        """Calculate circular standard deviation for angles"""
        angles = np.array(angles)
        angles_rad = np.radians(angles * 2)
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        R = np.sqrt(mean_cos**2 + mean_sin**2)

        if R < 1e-10:
            circular_std = 180.0
        else:
            circular_std = np.degrees(np.sqrt(-2 * np.log(R))) / 2
        return circular_std

    def check_if_rotation_needed(self, image: cv2.Mat) -> Tuple[bool, float]:
        """Check if rotation is needed and return rotation angle"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        min_votes = max(30, min(w, h) // 6)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min_votes)

        if lines is None or len(lines) < 2:
            return False, 0.0

        raw_angles = []
        for rho, theta in lines[:min(8, len(lines)), 0]:
            angle_deg = np.degrees(theta)
            if angle_deg > 90:
                angle_deg -= 180
            raw_angles.append(angle_deg)

        if not raw_angles:
            return False, 0.0

        rotation_angle, _ = self.determine_rotation_direction_and_angle(raw_angles)
        circular_std = self.calculate_circular_std(raw_angles)
        abs_rotation = abs(rotation_angle)

        # Border validation for small rotations
        if abs_rotation > 0.3 and abs_rotation < 15.0:
            border_width = min(h, w) // 12
            mask = np.zeros_like(gray)
            mask[:border_width, :] = 255
            mask[-border_width:, :] = 255
            mask[:, :border_width] = 255
            mask[:, -border_width:] = 255

            masked_edges = cv2.bitwise_and(edges, mask)
            min_votes_border = max(20, min(h, w) // 10)
            border_lines = cv2.HoughLines(masked_edges, 1, np.pi/180, threshold=min_votes_border)

            if border_lines is not None and len(border_lines) >= 3:
                border_angles = []
                for rho, theta in border_lines[:8, 0]:
                    angle_deg = np.degrees(theta)
                    if angle_deg > 90:
                        angle_deg -= 180
                    border_angles.append(angle_deg)

                horizontal_borders = [a for a in border_angles if abs(a) <= 15]
                vertical_borders = [a for a in border_angles if abs(abs(a) - 90) <= 15]

                if len(horizontal_borders) >= 1 and len(vertical_borders) >= 1:
                    h_median = np.median(horizontal_borders) if horizontal_borders else 0
                    v_median = np.median(vertical_borders) if vertical_borders else 90
                    border_rotation_h = abs(h_median)
                    border_rotation_v = abs(90 - abs(v_median))
                    max_border_rotation = max(border_rotation_h, border_rotation_v)

                    if max_border_rotation < 1.0:
                        return False, 0.0
                    elif max_border_rotation * 3 < abs_rotation:
                        return False, 0.0

        # Determine if rotation is needed based on angle magnitude and scatter
        if abs_rotation >= 80:
            return True, rotation_angle
        elif abs_rotation >= 5.0:
            if circular_std > 60.0:
                return False, rotation_angle
            else:
                return True, rotation_angle
        elif abs_rotation >= 1.5:
            if circular_std > 100.0:
                return False, rotation_angle
            else:
                return True, rotation_angle
        elif abs_rotation >= 0.3:
            if circular_std > 10.0:
                return False, rotation_angle
            else:
                return True, rotation_angle
        else:
            return False, rotation_angle

    def determine_rotation_direction_and_angle(self, angles):
        """Determine the best rotation direction and angle for ID card alignment"""
        angles = np.array(angles)
        horizontal_angles = []
        vertical_angles = []
        diagonal_angles = []

        for angle in angles:
            normalized = angle
            if normalized > 90:
                normalized -= 180
            elif normalized < -90:
                normalized += 180

            if abs(normalized) <= 15:
                horizontal_angles.append(normalized)
            elif abs(normalized) >= 75:
                vertical_angles.append(normalized)
            elif abs(abs(normalized) - 45) <= 15:
                diagonal_angles.append(normalized)
            else:
                horizontal_angles.append(normalized)

        h_median = np.median(horizontal_angles) if horizontal_angles else None
        v_median = np.median(vertical_angles) if vertical_angles else None
        d_median = np.median(diagonal_angles) if diagonal_angles else None

        rotation_needed = 0.0
        direction = "no rotation needed"

        # Detect 90° rotation based on diagonal dominance
        total_lines = len(horizontal_angles) + len(vertical_angles) + len(diagonal_angles)
        diagonal_ratio = len(diagonal_angles) / total_lines if total_lines > 0 else 0

        is_90_degree_rotation = (
            (diagonal_ratio > 0.7 and len(diagonal_angles) >= 5) or
            (len(diagonal_angles) >= 8 and len(diagonal_angles) > 2 * (len(horizontal_angles) + len(vertical_angles)))
        )

        if is_90_degree_rotation:
            if d_median is not None:
                if d_median < -30:
                    rotation_needed = 90.0
                    direction = "clockwise 90.0°"
                elif d_median > 30:
                    rotation_needed = -90.0
                    direction = "counter-clockwise 90.0°"
                else:
                    rotation_needed = 90.0
                    direction = "clockwise 90.0°"
        elif len(horizontal_angles) > len(vertical_angles) and len(horizontal_angles) >= 3:
            if h_median is not None and abs(h_median) < 15:
                rotation_needed = 90.0
                direction = "clockwise 90.0°"
            elif h_median is not None:
                rotation_needed = -h_median
                if rotation_needed > 0:
                    direction = f"clockwise {abs(rotation_needed):.2f}°"
                elif rotation_needed < 0:
                    direction = f"counter-clockwise {abs(rotation_needed):.2f}°"
        elif len(vertical_angles) > 0 and v_median is not None:
            if v_median > 45:
                rotation_needed = 90.0 - v_median
            elif v_median < -45:
                rotation_needed = -90.0 - v_median
            else:
                rotation_needed = -v_median

            if len(horizontal_angles) > 0 and len(vertical_angles) > 0:
                if abs(rotation_needed) < 2.0:
                    rotation_needed = 0.0
                    direction = "no rotation needed"

            if abs(rotation_needed) > 30:
                rotation_needed = 0.0
                direction = "no rotation needed"
            elif abs(rotation_needed) < 0.5:
                rotation_needed = 0.0
                direction = "no rotation needed"
            elif rotation_needed != 0.0:
                if rotation_needed > 0:
                    direction = f"clockwise {abs(rotation_needed):.2f}°"
                elif rotation_needed < 0:
                    direction = f"counter-clockwise {abs(rotation_needed):.2f}°"
        elif len(horizontal_angles) > 0 and h_median is not None:
            rotation_needed = -h_median
            if rotation_needed > 0:
                direction = f"clockwise {abs(rotation_needed):.2f}°"
            elif rotation_needed < 0:
                direction = f"counter-clockwise {abs(rotation_needed):.2f}°"

        return rotation_needed, direction

    def detect_rotation_angle_advanced(self, image: cv2.Mat) -> float:
        """Advanced rotation detection for small rotations"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        edges = cv2.Canny(gray, 30, 120)

        # Border edge detection
        mask = np.zeros_like(gray)
        border_width = min(h, w) // 12
        mask[:border_width, :] = 255
        mask[-border_width:, :] = 255
        mask[:, :border_width] = 255
        mask[:, -border_width:] = 255

        masked_edges = cv2.bitwise_and(edges, mask)
        min_votes = max(25, min(h, w) // 8)
        lines = cv2.HoughLines(masked_edges, 1, np.pi/180, threshold=min_votes)

        if lines is not None and len(lines) >= 2:
            angles = []
            for rho, theta in lines[:10, 0]:
                angle_deg = np.degrees(theta)
                if angle_deg > 90:
                    angle_deg -= 180
                angles.append(angle_deg)

            if angles:
                angles = np.array(angles)
                if len(angles) > 3:
                    mean_angle = np.mean(angles)
                    std_angle = np.std(angles)
                    if std_angle > 0:
                        filtered_angles = angles[abs(angles - mean_angle) <= 2 * std_angle]
                        if len(filtered_angles) > 0:
                            angles = filtered_angles

                return float(np.median(angles))

        # Fallback to general line detection
        min_votes_general = max(20, min(h, w) // 10)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min_votes_general)
        if lines is not None and len(lines) >= 2:
            angles = []
            for rho, theta in lines[:8, 0]:
                angle_deg = np.degrees(theta)
                if angle_deg > 90:
                    angle_deg -= 180
                angles.append(angle_deg)

            if angles:
                angles = np.array(angles)
                if len(angles) > 2:
                    q75, q25 = np.percentile(angles, [75, 25])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    angles = angles[(angles >= lower_bound) & (angles <= upper_bound)]

                if len(angles) > 0:
                    return float(np.median(angles))

        return 0.0

    def rotate_image(self, image: cv2.Mat, angle: float) -> cv2.Mat:
        """Rotate image by given angle"""
        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        return rotated

    def correct_rotation_smart(self, image: cv2.Mat) -> Tuple[cv2.Mat, float]:
        """Detect and correct image rotation"""
        needs_rotation, detected_angle = self.check_if_rotation_needed(image)

        if not needs_rotation:
            return image, 0.0

        precise_angle = self.detect_rotation_angle_advanced(image)

        if abs(precise_angle) > 0.1 and abs(precise_angle) < 15.0:
            final_angle = precise_angle
        else:
            final_angle = detected_angle

        if abs(final_angle) > 0.1:
            corrected_image = self.rotate_image(image, final_angle)
            return corrected_image, final_angle
        else:
            return image, 0.0

    def enhance_text_darkness(self, image: cv2.Mat) -> cv2.Mat:
        """Darken text regions for better digit detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        text_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        text_mask_color = cv2.merge([text_mask] * 3)
        darkened_image = image.copy()
        darkened_image[text_mask_color == 255] = (darkened_image[text_mask_color == 255] * 0.2).astype(np.uint8)
        return darkened_image

    def crop_id_card(self, image_path: str) -> Tuple[str, cv2.Mat, cv2.Mat, Dict]:
        """Stage 1: Use main model to detect and crop the ID card"""

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = self.main_model(image_path)[0]

        try:
            card_class_id = [k for k, v in self.main_model.names.items() if v == 'id'][0]
        except IndexError:
            raise ValueError("'id' class not found in main model!")

        card_boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        id_boxes = [(box, conf) for box, cls, conf in zip(card_boxes, class_ids, confidences)
                    if cls == card_class_id and conf >= self.confidence_thresholds['id']]

        if not id_boxes:
            raise ValueError("No ID card detected!")

        best_box, best_conf = max(id_boxes, key=lambda x: x[1])
        x1, y1, x2, y2 = map(int, best_box)
        cropped = image[y1:y2, x1:x2]

        # Save original cropped image
        original_cropped_path = image_path.replace('.jpg', '_cropped_original.jpg').replace('.png', '_cropped_original.png')
        cv2.imwrite(original_cropped_path, cropped)

        # Apply rotation correction
        corrected_cropped, rotation_angle = self.correct_rotation_smart(cropped)

        # Initialize rotation_corrected_path
        rotation_corrected_path = None

        # Save rotation-corrected image only if rotation was actually applied
        if abs(rotation_angle) > 0.1:
            rotation_corrected_path = image_path.replace('.jpg', '_rotation_corrected.jpg').replace('.png', '_rotation_corrected.png')
            cv2.imwrite(rotation_corrected_path, corrected_cropped)
        else:
            # If no rotation, just use the original cropped
            corrected_cropped = cropped

        # Apply text enhancement for digit detection
        enhanced_cropped = self.enhance_text_darkness(corrected_cropped)

        # Save final processed image
        cropped_path = image_path.replace('.jpg', '_cropped_id.jpg').replace('.png', '_cropped_id.png')
        cv2.imwrite(cropped_path, enhanced_cropped)

        detection_info = {
            'bbox': (x1, y1, x2, y2),
            'confidence': float(best_conf),
            'cropped_path': cropped_path,
            'original_cropped_path': original_cropped_path,
            'rotation_corrected_path': rotation_corrected_path,
            'rotation_corrected': rotation_corrected_path is not None,
            'rotation_angle': float(rotation_angle)
        }

        return cropped_path, corrected_cropped, enhanced_cropped, detection_info,original_cropped_path



    def detect_digits_yolo_only(self, cropped_id_path: str) -> List[Dict]:
        """Stage 2: Detect exactly 14 digits using YOLO only"""
        results = self.digits_model(cropped_id_path)[0]

        if len(results.boxes) == 0:
            return []

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        digit_detections = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf >= self.confidence_thresholds['digit']:
                digit_value = self.digit_class_mapping.get(cls_id, None)
                if digit_value is not None:
                    x1, y1, x2, y2 = map(int, box)
                    digit_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'digit_value': digit_value,
                        'x_center': (x1 + x2) / 2,
                        'y_center': (y1 + y2) / 2
                    })

        # Sort by position (left to right)
        digit_detections.sort(key=lambda x: x['x_center'])

        # Keep top 14 digits by confidence if more than 14 detected
        if len(digit_detections) > self.expected_digits:
            sorted_by_conf = sorted(digit_detections, key=lambda x: x['confidence'], reverse=True)
            top_digits = sorted_by_conf[:self.expected_digits]
            digit_detections = sorted(top_digits, key=lambda x: x['x_center'])

        return digit_detections

    def construct_national_id_simple(self, digit_detections: List[Dict]) -> Tuple[str, Dict]:
        """Construct 14-digit national ID from YOLO detections"""
        if not digit_detections:
            return "", {"method": "yolo_only", "status": "failed", "reason": "no_detections"}

        national_id = ''.join([d['digit_value'] for d in digit_detections])

        result_info = {
            "method": "yolo_only",
            "status": "success" if len(national_id) == self.expected_digits else "partial",
            "detected_digits": len(digit_detections),
            "expected_digits": self.expected_digits,
            "confidence_scores": [d['confidence'] for d in digit_detections],
            "avg_confidence": float(np.mean([d['confidence'] for d in digit_detections])),
            "min_confidence": float(np.min([d['confidence'] for d in digit_detections])),
            "max_confidence": float(np.max([d['confidence'] for d in digit_detections]))
        }

        return national_id, result_info

    def preprocess_for_text(self, image: cv2.Mat) -> cv2.Mat:
        """Standard preprocessing for Arabic text"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.3, beta=15)
        return enhanced


    # def manual_correction_interface(self, fields: Dict[str, str]) -> Dict[str, str]:
    #     """Interactive interface for manual corrections"""
    #     if not self.enable_manual_correction:
    #         return fields

    #     corrected_fields = fields.copy()

    #     print("\n" + "="*60)
    #     print("MANUAL CORRECTION MODE")
    #     print("="*60)
    #     print("Review and correct any OCR mistakes:")
    #     print("Press Enter to keep current value, or type new value to change")
    #     print("-"*60)

    #     field_names = {
    #         'national_id': 'National ID',
    #         'first_name': 'First Name',
    #         'last_name': 'Last Name',
    #         'address1': 'Address Line 1',
    #         'address2': 'Address Line 2'
    #     }

    #     for field_key, field_display in field_names.items():
    #         if field_key in corrected_fields and field_key != '_extraction_info':
    #             current_value = corrected_fields[field_key]

    #             if current_value:
    #                 print(f"\n{field_display}:")
    #                 print(f"Current: {current_value}")

    #                 # Show suggestions for common corrections
    #                 # suggestions = self._get_correction_suggestions(current_value, field_key)
    #                 # if suggestions:
    #                 #     print("Suggestions:")
    #                 #     for i, suggestion in enumerate(suggestions[:3], 1):
    #                 #         print(f"  {i}. {suggestion}")
    #                 #     print(f"  Or type your own correction:")

    #                 user_input = input(f"New value (or Enter to keep): ").strip()

    #                 if user_input:
    #                     # # Check if user selected a suggestion number
    #                     # if user_input.isdigit() and suggestions:
    #                     #     suggestion_idx = int(user_input) - 1
    #                     #     if 0 <= suggestion_idx < len(suggestions):
    #                     #         corrected_fields[field_key] = suggestions[suggestion_idx]
    #                     #         print(f"✅ Changed to: {suggestions[suggestion_idx]}")
    #                     #     else:
    #                     #         print("❌ Invalid suggestion number")

    #                     corrected_fields[field_key] = user_input
    #                     print(f"✅ Changed to: {user_input}")
    #                 else:
    #                     print("✅ Kept original value")
    #             else:
    #                 print(f"\n{field_display}: (Not detected)")
    #                 user_input = input(f"Enter value manually (or Enter to skip): ").strip()
    #                 if user_input:
    #                     corrected_fields[field_key] = user_input
    #                     print(f"✅ Added: {user_input}")

    #     print("\n" + "="*60)
    #     print("CORRECTION COMPLETED")
    #     print("="*60)

    #     return corrected_fields

    def extract_text_from_roi(self, image: cv2.Mat, bbox: Tuple[int, int, int, int], field_type: str) -> str:
        """Extract text from ROI for non-digit fields"""
        x1, y1, x2, y2 = bbox
        padding = 5
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return ""

        processed_roi = self.preprocess_for_text(roi)
        results = self.reader.readtext(processed_roi, detail=1, paragraph=True)

        confidence_threshold = self.confidence_thresholds.get(field_type, 0.4)
        best_text = ""

        for result in results:
            try:
                if len(result) == 3:
                    _, text, confidence = result
                elif len(result) == 2:
                    _, text = result
                    confidence = 1.0
                else:
                    continue

                if confidence >= confidence_threshold:
                    cleaned_text = text.strip()
                    if len(cleaned_text) > len(best_text):
                        best_text = cleaned_text
            except Exception:
                continue

        return best_text

    def detect_other_fields_from_processed_image(self, text_extraction_image: cv2.Mat, cropped_id_path: str) -> Dict[str, str]:
        """Extract other fields using the main model"""
        results = self.main_model(cropped_id_path)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        class_names = self.main_model.names

        detected_fields = {
            'first_name': '',
            'last_name': '',
            'address1': '',
            'address2': ''
        }

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            field_type = class_names[class_id]
            if field_type in detected_fields and conf >= self.confidence_thresholds.get(field_type, 0.4):
                x1, y1, x2, y2 = map(int, box)
                extracted_text = self.extract_text_from_roi(text_extraction_image, (x1, y1, x2, y2), field_type)
                if extracted_text:
                    detected_fields[field_type] = extracted_text

        return detected_fields

    def process_id_image(self, image_path: str) -> Dict[str, str]:
        """Complete processing pipeline"""
        # Stage 1: Crop ID card
        cropped_path, text_extraction_image, final_processed_image, crop_info,original_cropped_path = self.crop_id_card(image_path)

        # Stage 2: Extract national ID using YOLO
        digit_detections = self.detect_digits_yolo_only(cropped_path)
        national_id, extraction_info = self.construct_national_id_simple(digit_detections)

        # Stage 3: Extract other fields
        rotation_corrected_path = crop_info.get('rotation_corrected_path')
        if rotation_corrected_path:
            other_fields = self.detect_other_fields_from_processed_image(text_extraction_image, rotation_corrected_path)
        else:
            other_fields = self.detect_other_fields_from_processed_image(text_extraction_image, cropped_path)

        # Combine results
        final_results = {
            'national_id': national_id,
            **other_fields,
            '_extraction_info': extraction_info,
            'cropped_path': original_cropped_path,
            'final_processed_image': rotation_corrected_path,
        }
        # if self.enable_manual_correction:
        #     final_results = self.manual_correction_interface(final_results)

        return final_results

    def print_results(self, fields: Dict[str, str]):
        """Print formatted results"""
        print("\nArabic National ID Detection Results")
        print("=" * 50)

        for field_name, value in fields.items():
            if field_name.startswith('_'):
                continue
            print(f"{field_name.replace('_', ' ').title()}: {value if value else 'Not detected'}")

        if '_extraction_info' in fields:
            info = fields['_extraction_info']
            print(f"\nTechnical Summary:")
            print(f"Method: {info.get('method', 'unknown')}")
            print(f"Status: {info.get('status', 'unknown')}")
            print(f"Digits detected: {info.get('detected_digits', 0)}/{info.get('expected_digits', 14)}")
            print(f"Average confidence: {info.get('avg_confidence', 0):.2f}")