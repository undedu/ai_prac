from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from utils_ai import (
    process_image,
    process_video,
    generate_report,
    save_to_history,
    init_db,
    export_history_to_xlsx
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['REPORT_FOLDER'] = 'reports'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('media')
    if not file:
        return jsonify({'error': 'Файл не загружен'}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.mp4'):
        return jsonify({'error': 'Неподдерживаемый тип файла'}), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        if ext in ('.jpg', '.jpeg', '.png'):
            result = process_image(save_path)
            media_type = 'image'
        else:
            result = process_video(save_path)
            media_type = 'video'

        output_path = result['output_path']
        count = result['unique_people_sitting']

        # Перемещаем файл результата в static/
        static_result_name = os.path.basename(output_path)
        static_result_path = os.path.join(app.config['STATIC_FOLDER'], static_result_name)
        os.replace(output_path, static_result_path)

        report_path = generate_report(
            report_type="pdf",
            result_data={"filename": filename, "count": count},
            report_dir=app.config['REPORT_FOLDER']
        )
        report_name = os.path.basename(report_path)

        save_to_history(filename, media_type, count)

        return jsonify({
            "count": count,
            "output_url": f"/static/{static_result_name}",
            "report_url": f"/download/report/{report_name}",
            "media_type": media_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/report/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename)

@app.route('/download/history')
def download_history():
    path = export_history_to_xlsx()
    filename = os.path.basename(path)
    return send_from_directory(os.path.dirname(path), filename)

if __name__ == '__main__':
    app.run(debug=True)
