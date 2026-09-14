# วิธีรันบนเครื่องส่วนตัว

## สิ่งที่ต้องมี

- Python 3.11 หรือใหม่กว่า
- ระบบปฏิบัติการ Windows, macOS หรือ Linux

## 1. สร้าง virtual environment และติดตั้งไลบรารี

เปิด terminal ที่โฟลเดอร์โปรเจกต์ แล้วรัน:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

สำหรับ macOS/Linux ใช้คำสั่ง activate นี้แทน:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. รันเว็บแอป

```powershell
python app.py
```

เปิด [http://localhost:5000](http://localhost:5000) ในเบราว์เซอร์ แล้วอัปโหลด JPG หรือ PNG ขนาดไม่เกิน 10 MB

## ตรวจสอบเมื่อใช้งานไม่ได้

- หน้าเว็บแจ้งว่าไฟล์ไม่ถูกต้อง: ใช้ JPG หรือ PNG ที่อ่านได้ และมีขนาดไม่เกิน 10 MB
- ตรวจจับป้ายผิด: ใช้รูปที่คมขึ้น หรือปรับ 4 จุดมุมป้ายด้วยตนเอง
