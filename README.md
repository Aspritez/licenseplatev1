# Thai License Plate Deskew

เว็บแอป Streamlit สำหรับเลือกตำแหน่งป้ายทะเบียน 4 จุด ปรับมุมภาพป้ายด้วย Perspective Transform และเพิ่มความคมชัดด้วย OpenCV

แอปไม่มีฐานข้อมูล ไม่เก็บไฟล์อัปโหลด และเก็บจุดที่ผู้ใช้เลือกไว้เฉพาะระหว่างเปิดหน้าเว็บนั้น

## ความสามารถ

- อัปโหลดภาพ JPG หรือ PNG ขนาดไม่เกิน 10 MB
- ค้นหาตำแหน่งป้ายอัตโนมัติ หรือคลิกเลือกมุมป้าย 4 จุดด้วยตนเอง
- ป้องกันจุดซ้ำ รูปสี่เหลี่ยมผิดรูป และการตัดภาพที่ไม่ถูกต้อง
- แสดงภาพต้นฉบับพร้อมกรอบ และภาพป้ายที่ปรับมุมแล้ว
- ดาวน์โหลดภาพผลลัพธ์เป็น PNG

## Deploy บน Streamlit Community Cloud

1. Push โปรเจกต์นี้ขึ้น GitHub
2. เข้า [Streamlit Community Cloud](https://share.streamlit.io/) และเชื่อมบัญชี GitHub
3. เลือก repository, branch และไฟล์หลัก `app.py`
4. กด **Deploy**

Community Cloud จะติดตั้งไลบรารีจาก `requirements.txt` ให้อัตโนมัติ

## ไฟล์ที่จำเป็น

```text
app.py             Streamlit application
requirements.txt   dependencies for Streamlit Community Cloud
```
