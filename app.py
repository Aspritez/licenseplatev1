"""Thai License Plate Detection — Flask Application."""
from flask import Flask, render_template, request, jsonify
from utils import image_to_base64, base64_to_image, bytes_to_image, draw_corners_on_image, draw_all_candidates
from cv_pipeline import run_detection
from coordinate_transform import transform_plate

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload


@app.route('/')
def index():
    return render_template('index.html', upload_error=None)


@app.route('/process', methods=['POST'])
def process():
    # The upload form uses the "image" field. Accept the former "file" field
    # too, so a bookmarked older page does not fail unexpectedly.
    file = request.files.get('image') or request.files.get('file')
    if file is None:
        return render_template('index.html', upload_error='กรุณาเลือกไฟล์รูปภาพ'), 400

    if file.filename == '':
        return render_template('index.html', upload_error='กรุณาเลือกไฟล์รูปภาพ'), 400
    
    # Read file bytes
    file_bytes = file.read()
    
    # Validate image
    img = bytes_to_image(file_bytes)
    if img is None:
        return render_template(
            'index.html',
            upload_error='ไฟล์ที่อัปโหลดไม่ใช่รูปภาพที่รองรับ กรุณาเลือกไฟล์ JPG หรือ PNG'
        ), 400
    
    # Store original image as base64 for stateless operation
    original_b64 = image_to_base64(img)
    
    # Run CV detection pipeline
    detection = run_detection(file_bytes)
    
    # Prepare template variables
    template_vars = {
        'original_image_b64': original_b64,
        'annotated_image_b64': None,
        'warped_image_b64': None,
        'preprocess_confidence': detection.preprocess_confidence,
        'contour_confidence': detection.contour_confidence,
        'corner_confidence': detection.corner_confidence,
        'overall_confidence': detection.overall_detection_confidence,
        'corners': None,
        'error_code': detection.error_code,
        'error_message': detection.error_message,
        'deblur_applied': detection.deblur_applied,
        'deblur_method': detection.deblur_method,
        'blur_score_before': detection.blur_score_before,
        'blur_score_after': detection.blur_score_after,
        'still_blurry': detection.blur_score_after < 50,
        'has_multiple_candidates': False,
        'num_candidates': 0,
    }
    
    # Handle annotated image
    if detection.annotated_image is not None:
        template_vars['annotated_image_b64'] = image_to_base64(detection.annotated_image)
    
    # If detection succeeded, transform the detected plate region.
    if detection.success and detection.corners is not None:
        corners_list = detection.corners.tolist()
        template_vars['corners'] = corners_list
        
        # Run coordinate transform
        transform_result = transform_plate(img, detection.corners)
        
        if transform_result.success and transform_result.warped_image is not None:
            template_vars['warped_image_b64'] = image_to_base64(transform_result.warped_image)
        else:
            if transform_result.error_code:
                template_vars['error_code'] = transform_result.error_code
                template_vars['error_message'] = transform_result.error_message
                # Do not expose invalid automatic points in the editor.  A
                # valid centred quadrilateral is supplied below instead.
                template_vars['corners'] = None
    if template_vars['corners'] is None:
        # Always provide an editable, valid starting quadrilateral when
        # automatic detection did not yield usable corners.  This includes
        # both no-plate and no-corner outcomes.
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        offset_x, offset_y = min(w // 4, 200), min(h // 4, 100)
        template_vars['corners'] = [
            [cx - offset_x, cy - offset_y],
            [cx + offset_x, cy - offset_y],
            [cx + offset_x, cy + offset_y],
            [cx - offset_x, cy + offset_y]
        ]
    
    return render_template('result.html', **template_vars)


@app.route('/manual-correct', methods=['POST'])
def manual_correct():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'success': False, 'error_code': 'INVALID_REQUEST', 
                          'error_message': 'ข้อมูลไม่ถูกต้อง'}), 400
        
        original_b64 = data.get('original_image')
        points = data.get('points')
        
        if not original_b64 or not points:
            return jsonify({'success': False, 'error_code': 'INVALID_REQUEST',
                          'error_message': 'กรุณาส่งรูปภาพและพิกัดจุดมุม'}), 400
        
        if len(points) != 4:
            return jsonify({'success': False, 'error_code': 'INVALID_POINTS',
                          'error_message': 'ต้องระบุจุดมุมทั้ง 4 จุด'}), 400
        
        # Decode image
        import numpy as np
        img = base64_to_image(original_b64)
        if img is None:
            return jsonify({'success': False, 'error_code': 'INVALID_IMAGE',
                          'error_message': 'ไม่สามารถอ่านรูปภาพได้'}), 400
        
        corners = np.asarray(points, dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            return jsonify({'success': False, 'error_code': 'INVALID_POINTS',
                          'error_message': 'ต้องระบุพิกัดตัวเลขของจุดมุมทั้ง 4 จุด'}), 400
        
        # Run transform
        # Canvas points retain their TL, TR, BR, BL identities. Validate that
        # order before sorting so crossed manual edges are reported to the user.
        transform_result = transform_plate(img, corners, validate_input_order=True)
        
        if (
            not transform_result.success
            or transform_result.warped_image is None
            or transform_result.ordered_corners is None
        ):
            return jsonify({
                'success': False,
                'error_code': transform_result.error_code or 'INVALID_POINTS',
                'error_message': (
                    transform_result.error_message
                    or 'ไม่สามารถสร้างภาพป้ายทะเบียนจากจุดมุมที่ระบุได้'
                )
            })

        # Both values are known to be present after the guard above.
        annotated = draw_corners_on_image(img, transform_result.ordered_corners)
        return jsonify({
            'success': True,
            'warped_image_b64': image_to_base64(transform_result.warped_image),
            'annotated_image_b64': image_to_base64(annotated),
            # Return the exact order used by the perspective transform so the
            # editable overlay and the cropped image always describe the same
            # quadrilateral after a manual correction.
            'ordered_corners': transform_result.ordered_corners.tolist(),
            'error_code': None,
            'error_message': None
        })

    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'error_code': 'INVALID_POINTS',
            'error_message': 'พิกัดจุดมุมไม่ถูกต้อง'
        }), 400
    except Exception:
        return jsonify({
            'success': False,
            'error_code': 'SERVER_ERROR',
            'error_message': 'เกิดข้อผิดพลาดภายในระบบ กรุณาลองใหม่อีกครั้ง'
        }), 500


@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', upload_error='ไฟล์มีขนาดใหญ่เกินไป (สูงสุด 10MB)'), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
