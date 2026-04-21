import os, json, uuid, datetime
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, send_file)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, numpy as np
from utils.db import init_db, get_db, ensure_referrals_table
from utils.predictor import (load_models, predict_image,
                              load_class_names_from_json, load_manifest_info)
from utils.reports import generate_pdf_report

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'skindx-secret-2026-gctu')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DATABASE'] = 'skindx.db'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Class names ───────────────────────────────────────────────────────────────
_HARDCODED_CLASS_NAMES = [
    "Acne","Actinic_Keratosis","Benign_tumors","Bullous","Candidiasis",
    "DrugEruption","Eczema","Infestations_Bites","Lichen","Lupus","Moles",
    "Psoriasis","Rosacea","Seborrh_Keratoses","SkinCancer","Sun_Sunlight_Damage",
    "Tinea","Unknown_Normal","Vascular_Tumors","Vasculitis","Vitiligo","Warts"
]
_json_names = load_class_names_from_json(_HARDCODED_CLASS_NAMES)
CLASS_NAMES = _json_names if _json_names else _HARDCODED_CLASS_NAMES
print(f"Using {len(CLASS_NAMES)} class names")

# ── Recommendations ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Acne":{"severity":"Mild-Moderate","description":"Acne vulgaris is a common skin condition involving hair follicles and sebaceous glands.","recommendations":["Use non-comedogenic skincare products","Topical benzoyl peroxide or salicylic acid","Avoid touching or picking at lesions","Consult a dermatologist for persistent cases"],"urgency":"low","specialist":"Dermatologist"},
    "Actinic_Keratosis":{"severity":"Moderate-High","description":"Precancerous lesions caused by prolonged UV exposure requiring prompt evaluation.","recommendations":["Immediate dermatology referral recommended","Avoid sun exposure — use SPF 50+ sunscreen","Cryotherapy or topical treatment may be needed","Regular skin checks every 6 months"],"urgency":"high","specialist":"Dermatologist / Oncologist"},
    "Benign_tumors":{"severity":"Low","description":"Non-cancerous skin growths that are generally harmless but should be monitored.","recommendations":["Annual dermatology review","Monitor for changes in size, shape or colour","Surgical removal if causing discomfort","No urgent action required"],"urgency":"low","specialist":"Dermatologist"},
    "Bullous":{"severity":"High","description":"Blistering skin disorders that can be autoimmune in nature and require urgent evaluation.","recommendations":["Urgent dermatology referral","Avoid rupturing blisters","Keep affected areas clean and covered","Systemic treatment (steroids) may be required"],"urgency":"high","specialist":"Dermatologist"},
    "Candidiasis":{"severity":"Moderate","description":"Fungal infection caused by Candida species, common in warm, moist skin folds.","recommendations":["Antifungal cream (clotrimazole or miconazole)","Keep affected area dry and clean","Wear loose-fitting breathable clothing","Check for underlying diabetes or immunosuppression"],"urgency":"medium","specialist":"General Practitioner / Dermatologist"},
    "DrugEruption":{"severity":"Moderate-High","description":"Skin reaction caused by medication — may range from mild rash to severe reaction.","recommendations":["Identify and stop suspected causative drug immediately","Urgent medical evaluation","Antihistamines for mild reactions","Hospitalisation may be needed for severe cases (Stevens-Johnson)"],"urgency":"high","specialist":"General Practitioner / Emergency Medicine"},
    "Eczema":{"severity":"Mild-Moderate","description":"Chronic inflammatory skin condition causing dry, itchy, inflamed skin.","recommendations":["Moisturise frequently with emollients","Avoid known triggers (soaps, stress, allergens)","Topical corticosteroids for flares","Consult dermatologist for severe or recurrent cases"],"urgency":"low","specialist":"Dermatologist / Allergist"},
    "Infestations_Bites":{"severity":"Mild-Moderate","description":"Skin reactions from insect bites or parasitic infestations such as scabies or lice.","recommendations":["Permethrin or ivermectin for scabies","Treat all close contacts simultaneously","Wash clothing and bedding at high temperature","Antihistamines for itch relief"],"urgency":"medium","specialist":"General Practitioner"},
    "Lichen":{"severity":"Moderate","description":"Lichen planus is an inflammatory condition affecting skin and mucous membranes.","recommendations":["Topical corticosteroids for symptom relief","Avoid scratching affected areas","Dermatology referral for extensive disease","Check for associated hepatitis C infection"],"urgency":"medium","specialist":"Dermatologist"},
    "Lupus":{"severity":"High","description":"Systemic autoimmune disease with skin manifestations including the butterfly rash.","recommendations":["Urgent rheumatology / dermatology referral","Sun protection is critical (SPF 50+)","Systemic evaluation for organ involvement","Long-term specialist monitoring required"],"urgency":"high","specialist":"Rheumatologist / Dermatologist"},
    "Moles":{"severity":"Low-Moderate","description":"Pigmented lesions that are usually benign but require monitoring using ABCDE criteria.","recommendations":["Apply ABCDE rule: Asymmetry, Border, Colour, Diameter, Evolution","Annual skin examination","Dermatoscopy evaluation if changing","Excision if atypical features present"],"urgency":"low","specialist":"Dermatologist"},
    "Psoriasis":{"severity":"Moderate","description":"Chronic autoimmune condition causing rapid skin cell build-up with scaly patches.","recommendations":["Topical corticosteroids and vitamin D analogues","Phototherapy for moderate-severe disease","Avoid stress, alcohol and smoking","Dermatology referral for systemic treatment options"],"urgency":"medium","specialist":"Dermatologist"},
    "Rosacea":{"severity":"Mild-Moderate","description":"Chronic skin condition causing redness and visible blood vessels on the face.","recommendations":["Avoid triggers: heat, spicy food, alcohol, sun","Topical metronidazole or azelaic acid","Gentle non-irritating skincare routine","Dermatology referral for oral antibiotics if needed"],"urgency":"low","specialist":"Dermatologist"},
    "Seborrh_Keratoses":{"severity":"Low","description":"Common benign skin growths appearing as waxy, stuck-on lesions.","recommendations":["No treatment required for asymptomatic lesions","Cryotherapy or curettage if bothersome","Monitor for changes in appearance","Reassurance — not precancerous"],"urgency":"low","specialist":"Dermatologist (optional)"},
    "SkinCancer":{"severity":"Critical","description":"Malignant skin neoplasm requiring immediate specialist evaluation and biopsy.","recommendations":["URGENT: Immediate oncology/dermatology referral","Do not delay — biopsy and histology needed","Avoid further sun exposure","Complete skin examination for additional lesions"],"urgency":"critical","specialist":"Dermatologist / Oncologist"},
    "Sun_Sunlight_Damage":{"severity":"Moderate","description":"Photoaging and skin damage from chronic ultraviolet radiation exposure.","recommendations":["Daily SPF 50+ broad-spectrum sunscreen","Protective clothing and hat outdoors","Topical retinoids to reverse photoaging","Annual skin cancer screening"],"urgency":"medium","specialist":"Dermatologist"},
    "Tinea":{"severity":"Mild-Moderate","description":"Fungal infection (ringworm/athlete's foot) caused by dermatophytes.","recommendations":["Topical antifungal (clotrimazole, terbinafine)","Keep area dry; change socks daily for tinea pedis","Oral antifungals for extensive or nail disease","Avoid sharing personal items"],"urgency":"low","specialist":"General Practitioner"},
    "Unknown_Normal":{"severity":"None","description":"No significant skin pathology detected. Skin appears within normal limits.","recommendations":["Maintain regular skincare routine","Annual skin health check-up","Apply sunscreen daily","Report any new or changing lesions promptly"],"urgency":"none","specialist":"No referral required"},
    "Vascular_Tumors":{"severity":"Moderate","description":"Benign or malignant vascular neoplasms including haemangiomas and Kaposi sarcoma.","recommendations":["Dermatology referral for classification","Imaging may be needed for deeper lesions","Treatment depends on type (laser, surgery, chemo)","Monitor for rapid growth"],"urgency":"medium","specialist":"Dermatologist / Vascular Surgeon"},
    "Vasculitis":{"severity":"High","description":"Inflammation of blood vessel walls causing skin and systemic manifestations.","recommendations":["Urgent medical evaluation — systemic involvement likely","Blood tests: CBC, ESR, CRP, ANCA","Rheumatology referral","Corticosteroids and immunosuppressants may be needed"],"urgency":"high","specialist":"Rheumatologist / Dermatologist"},
    "Vitiligo":{"severity":"Low-Moderate","description":"Autoimmune condition causing loss of skin pigmentation in patches.","recommendations":["Sun protection critical (unpigmented skin burns easily)","Topical corticosteroids or calcineurin inhibitors","Phototherapy (NB-UVB) for repigmentation","Psychological support — significant quality of life impact"],"urgency":"low","specialist":"Dermatologist"},
    "Warts":{"severity":"Low","description":"Benign viral skin growths caused by Human Papillomavirus (HPV).","recommendations":["Salicylic acid topical treatment","Cryotherapy by healthcare provider","Avoid picking — prevents spread","Most resolve spontaneously within 2 years"],"urgency":"low","specialist":"General Practitioner"},
}

# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            if user['status'] == 'pending':
                flash('Your account is pending administrator approval.', 'error')
                return redirect(url_for('login'))
            if user['status'] == 'suspended':
                flash('Your account has been suspended. Contact admin.', 'error')
                return redirect(url_for('login'))
            session.update({'user_id':user['id'],'username':user['username'],
                            'role':user['role'],'full_name':user['full_name']})
            db.execute('UPDATE users SET last_login=? WHERE id=?',
                       (datetime.datetime.now().isoformat(), user['id']))
            db.commit()
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username  = request.form.get('username','').strip()
        full_name = request.form.get('full_name','').strip()
        staff_id  = request.form.get('staff_id','').strip()
        email     = request.form.get('email','').strip()
        department= request.form.get('department','').strip()
        password  = request.form.get('password','')
        confirm   = request.form.get('confirm_password','')
        if not all([username, full_name, password]):
            flash('Username, full name and password are required.', 'error')
            return redirect(url_for('signup'))
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('signup'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('signup'))
        db = get_db()
        try:
            db.execute(
                """INSERT INTO users
                   (username,full_name,password_hash,role,status,staff_id,email,department,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (username, full_name, generate_password_hash(password),
                 'user', 'pending', staff_id, email, department,
                 datetime.datetime.now().isoformat()))
            db.commit()
            flash('Registration submitted! Please wait for administrator approval before logging in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken. Please choose another.', 'error')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    is_admin = session['role'] == 'admin'

    if is_admin:
        total_predictions = db.execute('SELECT COUNT(*) FROM predictions WHERE is_deleted=0').fetchone()[0]
        today = datetime.date.today().isoformat()
        today_predictions = db.execute(
            'SELECT COUNT(*) FROM predictions WHERE is_deleted=0 AND date(created_at)=?',(today,)).fetchone()[0]
        total_users   = db.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
        pending_users = db.execute("SELECT COUNT(*) FROM users WHERE status='pending'").fetchone()[0]
        recent = db.execute(
            '''SELECT p.*,u.full_name FROM predictions p
               JOIN users u ON p.user_id=u.id
               WHERE p.is_deleted=0
               ORDER BY p.created_at DESC LIMIT 8''').fetchall()
        class_dist = db.execute(
            '''SELECT predicted_class,COUNT(*) as cnt FROM predictions
               WHERE is_deleted=0 GROUP BY predicted_class ORDER BY cnt DESC LIMIT 10''').fetchall()
        weekly = db.execute(
            '''SELECT date(created_at) as day,COUNT(*) as cnt FROM predictions
               WHERE is_deleted=0 AND created_at>=date('now','-7 days')
               GROUP BY day ORDER BY day''').fetchall()
        urgency_counts = db.execute(
            '''SELECT urgency_level,COUNT(*) as cnt FROM predictions
               WHERE is_deleted=0 GROUP BY urgency_level''').fetchall()
    else:
        uid = session['user_id']
        total_predictions = db.execute(
            'SELECT COUNT(*) FROM predictions WHERE user_id=? AND is_deleted=0',(uid,)).fetchone()[0]
        today = datetime.date.today().isoformat()
        today_predictions = db.execute(
            'SELECT COUNT(*) FROM predictions WHERE user_id=? AND is_deleted=0 AND date(created_at)=?',
            (uid, today)).fetchone()[0]
        total_users   = 0
        pending_users = 0
        recent = db.execute(
            '''SELECT p.*,u.full_name FROM predictions p
               JOIN users u ON p.user_id=u.id
               WHERE p.user_id=? AND p.is_deleted=0
               ORDER BY p.created_at DESC LIMIT 8''',(uid,)).fetchall()
        class_dist = db.execute(
            '''SELECT predicted_class,COUNT(*) as cnt FROM predictions
               WHERE user_id=? AND is_deleted=0
               GROUP BY predicted_class ORDER BY cnt DESC LIMIT 10''',(uid,)).fetchall()
        weekly = db.execute(
            '''SELECT date(created_at) as day,COUNT(*) as cnt FROM predictions
               WHERE user_id=? AND is_deleted=0 AND created_at>=date('now','-7 days')
               GROUP BY day ORDER BY day''',(uid,)).fetchall()
        urgency_counts = db.execute(
            '''SELECT urgency_level,COUNT(*) as cnt FROM predictions
               WHERE user_id=? AND is_deleted=0 GROUP BY urgency_level''',(uid,)).fetchall()

    # Referral notifications for health workers
    my_approved_referrals = 0
    my_pending_referrals  = 0
    if not is_admin:
        ensure_referrals_table()
        my_approved_referrals = db.execute(
            """SELECT COUNT(*) FROM referrals
               WHERE requester_id=? AND status='approved'""",
            (session['user_id'],)).fetchone()[0]
        my_pending_referrals = db.execute(
            """SELECT COUNT(*) FROM referrals
               WHERE requester_id=? AND status='pending'""",
            (session['user_id'],)).fetchone()[0]

    return render_template('dashboard.html',
        total_predictions=total_predictions,
        today_predictions=today_predictions,
        total_users=total_users,
        pending_users=pending_users,
        recent=recent,
        class_dist=[dict(r) for r in class_dist],
        weekly=[dict(r) for r in weekly],
        urgency_counts=[dict(r) for r in urgency_counts],
        is_admin=is_admin,
        my_approved_referrals=my_approved_referrals,
        my_pending_referrals=my_pending_referrals)

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded.', 'error')
            return redirect(request.url)
        file = request.files['image']
        patient_name = request.form.get('patient_name','Anonymous').strip()
        patient_age  = request.form.get('patient_age','').strip()
        patient_sex  = request.form.get('patient_sex','').strip()
        notes        = request.form.get('notes','').strip()
        if file.filename == '' or not allowed_file(file.filename):
            flash('Please upload a valid image (JPG, PNG, BMP, WEBP).', 'error')
            return redirect(request.url)
        ext   = file.filename.rsplit('.',1)[1].lower()
        fname = f"{uuid.uuid4().hex}.{ext}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)
        result     = predict_image(fpath, CLASS_NAMES)
        pred_class = result['predicted_class']
        confidence = result['confidence']
        top5       = result['top5']
        rec        = RECOMMENDATIONS.get(pred_class, {})
        # Store UUID so result page can link back to exact record
        record_id = str(uuid.uuid4())
        db = get_db()
        db.execute(
            '''INSERT INTO predictions
               (id,user_id,patient_name,patient_age,patient_sex,
                image_path,predicted_class,confidence,top5_json,
                urgency_level,notes,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (record_id, session['user_id'], patient_name,
             patient_age, patient_sex, fname, pred_class,
             round(confidence,4), json.dumps(top5),
             rec.get('urgency','low'), notes,
             datetime.datetime.now().isoformat()))
        db.commit()
        return render_template('result.html',
            pred_class=pred_class, confidence=confidence, top5=top5, rec=rec,
            patient_name=patient_name, patient_age=patient_age,
            patient_sex=patient_sex, notes=notes,
            image_path=fname, rid=record_id)
    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    db   = get_db()
    page = int(request.args.get('page',1))
    per_page = 12
    offset   = (page-1)*per_page
    search   = request.args.get('search','').strip()
    urgency_filter = request.args.get('urgency','').strip()
    is_admin = session['role'] == 'admin'

    query  = 'SELECT p.*,u.full_name FROM predictions p JOIN users u ON p.user_id=u.id WHERE p.is_deleted=0'
    params = []
    if not is_admin:
        query += ' AND p.user_id=?'
        params.append(session['user_id'])
    if search:
        query += ' AND (p.patient_name LIKE ? OR p.predicted_class LIKE ?)'
        params.extend([f'%{search}%', f'%{search}%'])
    if urgency_filter:
        query += ' AND p.urgency_level=?'
        params.append(urgency_filter)

    total = db.execute(f'SELECT COUNT(*) FROM ({query})', params).fetchone()[0]
    records = db.execute(query + ' ORDER BY p.created_at DESC LIMIT ? OFFSET ?',
                         params + [per_page, offset]).fetchall()
    total_pages = (total + per_page - 1) // per_page
    return render_template('history.html', records=records,
        page=page, total_pages=total_pages,
        search=search, urgency_filter=urgency_filter,
        total=total, is_admin=is_admin)

@app.route('/record/delete/<string:rid>', methods=['POST'])
@login_required
def delete_record(rid):
    db     = get_db()
    reason = request.form.get('reason','').strip()
    record = db.execute('SELECT * FROM predictions WHERE id=?',(rid,)).fetchone()
    if not record:
        flash('Record not found.', 'error')
        return redirect(url_for('history'))
    # Health workers can only delete their own records
    if session['role'] != 'admin' and record['user_id'] != session['user_id']:
        flash('You can only delete your own records.', 'error')
        return redirect(url_for('history'))
    db.execute(
        '''UPDATE predictions SET is_deleted=1, deleted_by=?, deleted_at=?, delete_reason=?
           WHERE id=?''',
        (session['user_id'], datetime.datetime.now().isoformat(), reason, rid))
    db.commit()
    flash('Record deleted successfully.', 'success')
    return redirect(url_for('history'))

@app.route('/report/<string:rid>')
@login_required
def view_report(rid):
    db = get_db()
    pred = db.execute(
        'SELECT p.*,u.full_name,u.username FROM predictions p JOIN users u ON p.user_id=u.id WHERE p.id=?',
        (rid,)).fetchone()
    if not pred:
        flash('Record not found.', 'error')
        return redirect(url_for('history'))
    is_owner = pred['user_id'] == session['user_id']
    is_admin = session['role'] == 'admin'
    if not is_owner and not is_admin:
        # Check approved referral
        approved = db.execute(
            """SELECT id FROM referrals
               WHERE record_id=? AND requester_id=? AND status='approved'""",
            (rid, session['user_id'])).fetchone()
        if not approved:
            # Redirect to request access page instead of hard block
            flash('You need access permission to view this record.', 'error')
            return redirect(url_for('request_access', rid=rid))
    rec  = RECOMMENDATIONS.get(pred['predicted_class'], {})
    top5 = json.loads(pred['top5_json']) if pred['top5_json'] else []
    return render_template('report.html', pred=pred, rec=rec, top5=top5)

@app.route('/report/<string:rid>/pdf')
@login_required
def download_report(rid):
    db = get_db()
    pred = db.execute(
        'SELECT p.*,u.full_name FROM predictions p JOIN users u ON p.user_id=u.id WHERE p.id=?',
        (rid,)).fetchone()
    if not pred:
        flash('Record not found.', 'error')
        return redirect(url_for('history'))
    is_owner = pred['user_id'] == session['user_id']
    is_admin = session['role'] == 'admin'
    if not is_owner and not is_admin:
        approved = db.execute(
            """SELECT id FROM referrals
               WHERE record_id=? AND requester_id=? AND status='approved'""",
            (rid, session['user_id'])).fetchone()
        if not approved:
            flash('You need access permission to download this report.', 'error')
            return redirect(url_for('request_access', rid=rid))
    rec  = RECOMMENDATIONS.get(pred['predicted_class'], {})
    top5 = json.loads(pred['top5_json']) if pred['top5_json'] else []
    pdf_path = generate_pdf_report(dict(pred), rec, top5)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f"SkinDx_Report_{rid[:8]}.pdf")

@app.route('/analytics')
@login_required
def analytics():
    db = get_db()
    is_admin = session['role'] == 'admin'
    uid = session['user_id']

    base = 'FROM predictions WHERE is_deleted=0'
    filt = '' if is_admin else f' AND user_id={uid}'

    class_dist = db.execute(
        f'SELECT predicted_class,COUNT(*) as cnt {base}{filt} GROUP BY predicted_class ORDER BY cnt DESC').fetchall()
    monthly = db.execute(
        f'''SELECT strftime("%Y-%m",created_at) as month,COUNT(*) as cnt
            {base}{filt} GROUP BY month ORDER BY month DESC LIMIT 12''').fetchall()
    urgency_dist = db.execute(
        f'SELECT urgency_level,COUNT(*) as cnt {base}{filt} GROUP BY urgency_level').fetchall()
    avg_conf = db.execute(f'SELECT AVG(confidence) {base}{filt}').fetchone()[0] or 0
    high_conf = db.execute(f'SELECT COUNT(*) {base}{filt} AND confidence>=0.80').fetchone()[0]

    top_users = []
    if is_admin:
        top_users = db.execute(
            '''SELECT u.full_name,COUNT(*) as cnt FROM predictions p
               JOIN users u ON p.user_id=u.id
               WHERE p.is_deleted=0
               GROUP BY p.user_id ORDER BY cnt DESC LIMIT 5''').fetchall()

    return render_template('analytics.html',
        class_dist=[dict(r) for r in class_dist],
        monthly=[dict(r) for r in monthly],
        urgency_dist=[dict(r) for r in urgency_dist],
        top_users=[dict(r) for r in top_users],
        avg_conf=round(avg_conf*100,1),
        high_conf=high_conf,
        is_admin=is_admin)

# ── Admin routes ──────────────────────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin():
    ensure_referrals_table()
    db = get_db()
    users        = db.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    total_preds  = db.execute('SELECT COUNT(*) FROM predictions WHERE is_deleted=0').fetchone()[0]
    total_users  = db.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
    pending      = db.execute("SELECT * FROM users WHERE status='pending' ORDER BY created_at DESC").fetchall()
    deleted_recs = db.execute(
        '''SELECT p.*,u.full_name as examined_by,
                  d.full_name as deleted_by_name
           FROM predictions p
           JOIN users u ON p.user_id=u.id
           LEFT JOIN users d ON p.deleted_by=d.id
           WHERE p.is_deleted=1
           ORDER BY p.deleted_at DESC LIMIT 50''').fetchall()
    referral_requests = db.execute(
        '''SELECT r.*, p.patient_name, p.predicted_class, p.id as record_id,
                  req.full_name as requester_name, req.staff_id as requester_staff_id,
                  own.full_name as owner_name
           FROM referrals r
           JOIN predictions p  ON r.record_id    = p.id
           JOIN users req      ON r.requester_id  = req.id
           JOIN users own      ON r.owner_id      = own.id
           WHERE r.status = 'pending'
           ORDER BY r.created_at DESC''').fetchall()
    all_referrals = db.execute(
        '''SELECT r.*, p.patient_name, p.predicted_class,
                  req.full_name as requester_name,
                  own.full_name as owner_name,
                  rev.full_name as reviewer_name
           FROM referrals r
           JOIN predictions p  ON r.record_id    = p.id
           JOIN users req      ON r.requester_id  = req.id
           JOIN users own      ON r.owner_id      = own.id
           LEFT JOIN users rev ON r.reviewed_by   = rev.id
           WHERE r.status != 'pending'
           ORDER BY r.reviewed_at DESC LIMIT 30''').fetchall()
    return render_template('admin.html', users=users,
        total_preds=total_preds, total_users=total_users,
        pending=pending, deleted_recs=deleted_recs,
        referral_requests=referral_requests,
        all_referrals=all_referrals)

@app.route('/admin/user/approve/<int:uid>', methods=['POST'])
@admin_required
def approve_user(uid):
    db = get_db()
    db.execute("UPDATE users SET status='active' WHERE id=?", (uid,))
    db.commit()
    flash('User account approved and activated.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/user/reject/<int:uid>', methods=['POST'])
@admin_required
def reject_user(uid):
    db = get_db()
    db.execute("DELETE FROM users WHERE id=? AND status='pending'", (uid,))
    db.commit()
    flash('Registration request rejected and removed.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/user/suspend/<int:uid>', methods=['POST'])
@admin_required
def suspend_user(uid):
    if uid == session['user_id']:
        flash('Cannot suspend your own account.', 'error')
        return redirect(url_for('admin'))
    db = get_db()
    db.execute("UPDATE users SET status='suspended' WHERE id=?", (uid,))
    db.commit()
    flash('User suspended.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/user/activate/<int:uid>', methods=['POST'])
@admin_required
def activate_user(uid):
    db = get_db()
    db.execute("UPDATE users SET status='active' WHERE id=?", (uid,))
    db.commit()
    flash('User reactivated.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/user/add', methods=['POST'])
@admin_required
def add_user():
    username  = request.form.get('username','').strip()
    full_name = request.form.get('full_name','').strip()
    staff_id  = request.form.get('staff_id','').strip()
    password  = request.form.get('password','')
    role      = request.form.get('role','user')
    if not all([username, full_name, password]):
        flash('All fields are required.', 'error')
        return redirect(url_for('admin'))
    db = get_db()
    try:
        db.execute(
            'INSERT INTO users (username,full_name,password_hash,role,status,staff_id,created_at) VALUES (?,?,?,?,?,?,?)',
            (username, full_name, generate_password_hash(password),
             role, 'active', staff_id, datetime.datetime.now().isoformat()))
        db.commit()
        flash(f'User {username} created successfully.', 'success')
    except sqlite3.IntegrityError:
        flash('Username already exists.', 'error')
    return redirect(url_for('admin'))

@app.route('/admin/user/delete/<int:uid>', methods=['POST'])
@admin_required
def delete_user(uid):
    if uid == session['user_id']:
        flash('Cannot delete your own account.', 'error')
        return redirect(url_for('admin'))
    db = get_db()
    db.execute('DELETE FROM users WHERE id=?', (uid,))
    db.commit()
    flash('User deleted.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/record/restore/<string:rid>', methods=['POST'])
@admin_required
def restore_record(rid):
    db = get_db()
    db.execute('UPDATE predictions SET is_deleted=0,deleted_by=NULL,deleted_at=NULL,delete_reason=NULL WHERE id=?',(rid,))
    db.commit()
    flash('Record restored successfully.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/model-info')
@admin_required
def model_info():
    manifest    = load_manifest_info()
    all_results = manifest.get('all_results', {})
    model_stats = [{
        'name':     name,
        'accuracy': round(r.get('test_accuracy',0)*100, 2),
        'f1':       round(r.get('macro_f1',0), 4),
        'auc':      round(r.get('auc_roc',0), 4),
        'params_m': round(r.get('params',0)/1e6, 2),
        'size_mb':  round(r.get('size_mb',0), 2),
        'infer_ms': round(r.get('inference_mean_ms',0), 2),
    } for name, r in all_results.items()]
    info = {
        'models':['EfficientNetB3','MobileNetV2','ResNet50'],
        'ensemble':'Triple-Branch Feature Concatenation (4,864-dim)',
        'classes': manifest.get('num_classes', len(CLASS_NAMES)),
        'img_size':'×'.join(str(s) for s in manifest.get('img_size',[224,224])),
        'framework':'TensorFlow 2.15 / Keras',
        'dataset':'Kaggle Skin Disease Dataset (22 classes, ~15,400 images)',
        'best_model':    manifest.get('best_model_name','EfficientNetB3'),
        'best_accuracy': round(manifest.get('test_accuracy',0)*100, 2),
        'best_f1':       round(manifest.get('macro_f1',0), 4),
        'best_auc':      round(manifest.get('auc_roc',0), 4),
        'model_stats':   model_stats,
        'manifest_loaded': bool(manifest),
    }
    return render_template('model_info.html', info=info)

@app.route('/api/stats')
@login_required
def api_stats():
    db = get_db()
    is_admin = session['role'] == 'admin'
    uid = session['user_id']
    base = 'FROM predictions WHERE is_deleted=0'
    filt = '' if is_admin else f' AND user_id={uid}'
    weekly = db.execute(
        f'''SELECT date(created_at) as day,COUNT(*) as cnt
            {base}{filt} AND created_at>=date('now','-7 days')
            GROUP BY day ORDER BY day''').fetchall()
    class_dist = db.execute(
        f'SELECT predicted_class,COUNT(*) as cnt {base}{filt} GROUP BY predicted_class ORDER BY cnt DESC LIMIT 8'
    ).fetchall()
    return jsonify({
        'weekly':     [{'day':r['day'],'cnt':r['cnt']} for r in weekly],
        'class_dist': [{'cls':r['predicted_class'],'cnt':r['cnt']} for r in class_dist]
    })

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    db = get_db()
    if request.method == 'POST':
        full_name    = request.form.get('full_name','').strip()
        staff_id     = request.form.get('staff_id','').strip()
        email        = request.form.get('email','').strip()
        department   = request.form.get('department','').strip()
        current_pass = request.form.get('current_password','')
        new_pass     = request.form.get('new_password','')
        confirm_pass = request.form.get('confirm_password','')

        user = db.execute('SELECT * FROM users WHERE id=?',(session['user_id'],)).fetchone()

        # Update profile fields
        if full_name:
            db.execute('UPDATE users SET full_name=?,staff_id=?,email=?,department=? WHERE id=?',
                       (full_name, staff_id, email, department, session['user_id']))
            session['full_name'] = full_name
            flash('Profile updated successfully.', 'success')

        # Change password
        if current_pass or new_pass:
            if not current_pass:
                flash('Enter your current password to change it.', 'error')
            elif not check_password_hash(user['password_hash'], current_pass):
                flash('Current password is incorrect.', 'error')
            elif len(new_pass) < 6:
                flash('New password must be at least 6 characters.', 'error')
            elif new_pass != confirm_pass:
                flash('New passwords do not match.', 'error')
            else:
                db.execute('UPDATE users SET password_hash=? WHERE id=?',
                           (generate_password_hash(new_pass), session['user_id']))
                flash('Password changed successfully.', 'success')
        db.commit()

    user     = db.execute('SELECT * FROM users WHERE id=?',(session['user_id'],)).fetchone()
    my_preds = db.execute(
        'SELECT COUNT(*) FROM predictions WHERE user_id=? AND is_deleted=0',(session['user_id'],)).fetchone()[0]
    return render_template('profile.html', user=user, my_preds=my_preds)

if __name__ == '__main__':
    with app.app_context():
        init_db()
        ensure_referrals_table()
    app.run(debug=True, host='0.0.0.0', port=5000)

# ── Referral / Access Request routes ─────────────────────────────────────────

@app.route('/record/request-access/<string:rid>', methods=['GET', 'POST'])
@login_required
def request_access(rid):
    """Health worker requests access to another worker's record."""
    db = get_db()
    pred = db.execute(
        '''SELECT p.*, u.full_name as owner_name
           FROM predictions p JOIN users u ON p.user_id=u.id
           WHERE p.id=? AND p.is_deleted=0''', (rid,)).fetchone()
    if not pred:
        flash('Record not found.', 'error')
        return redirect(url_for('history'))

    # Already the owner or admin - just view it directly
    if pred['user_id'] == session['user_id'] or session['role'] == 'admin':
        return redirect(url_for('view_report', rid=rid))

    # Check if already approved
    approved = db.execute(
        """SELECT * FROM referrals
           WHERE record_id=? AND requester_id=? AND status='approved'""",
        (rid, session['user_id'])).fetchone()
    if approved:
        return redirect(url_for('view_report', rid=rid))

    # Check if already pending
    pending = db.execute(
        """SELECT * FROM referrals
           WHERE record_id=? AND requester_id=? AND status='pending'""",
        (rid, session['user_id'])).fetchone()

    if request.method == 'POST':
        reason = request.form.get('reason', '').strip()
        if not reason:
            flash('Please provide a reason for the access request.', 'error')
            return redirect(url_for('request_access', rid=rid))
        if pending:
            flash('You already have a pending request for this record.', 'error')
            return redirect(url_for('history'))
        db.execute(
            """INSERT INTO referrals
               (record_id, requester_id, owner_id, reason, status, created_at)
               VALUES (?,?,?,?,?,?)""",
            (rid, session['user_id'], pred['user_id'], reason,
             'pending', datetime.datetime.now().isoformat()))
        db.commit()
        flash('Access request submitted. You will be notified once the administrator reviews it.', 'success')
        return redirect(url_for('history'))

    return render_template('request_access.html',
                           pred=pred, pending=pending, rid=rid)


@app.route('/admin/referral/approve/<int:ref_id>', methods=['POST'])
@admin_required
def approve_referral(ref_id):
    db = get_db()
    db.execute(
        """UPDATE referrals
           SET status='approved', reviewed_by=?, reviewed_at=?
           WHERE id=?""",
        (session['user_id'], datetime.datetime.now().isoformat(), ref_id))
    db.commit()
    flash('Access request approved. The health worker can now view this record.', 'success')
    return redirect(url_for('admin') + '#tab-referrals')


@app.route('/admin/referral/deny/<int:ref_id>', methods=['POST'])
@admin_required
def deny_referral(ref_id):
    db = get_db()
    db.execute(
        """UPDATE referrals
           SET status='denied', reviewed_by=?, reviewed_at=?
           WHERE id=?""",
        (session['user_id'], datetime.datetime.now().isoformat(), ref_id))
    db.commit()
    flash('Access request denied.', 'success')
    return redirect(url_for('admin') + '#tab-referrals')

