# วิธี Build และ Deploy ด้วย Docker

Dockerfile ของโปรเจกต์ติดตั้ง Python และ OpenCV ไว้แล้ว จึงเป็นวิธี deploy ที่แนะนำ

## 1. Build image

รันจากโฟลเดอร์โปรเจกต์:

```bash
docker build -t thai-license-plate-ocr:latest .
```

## 2. รัน container บนเครื่อง

```bash
docker run --rm -p 5000:5000 \
  --name thai-license-plate-ocr \
  thai-license-plate-ocr:latest
```

จากนั้นเปิด `http://localhost:5000`

สำหรับ PowerShell ให้ใช้บรรทัดเดียว:

```powershell
docker run --rm -p 5000:5000 --name thai-license-plate-ocr thai-license-plate-ocr:latest
```

## 3. Deploy บนเซิร์ฟเวอร์

1. ติดตั้ง Docker และเปิด firewall เฉพาะ port ที่ reverse proxy ใช้งาน
2. คัดลอก source code ไปยังเซิร์ฟเวอร์ แล้ว build image ตามขั้นตอนที่ 1
3. รัน container หลัง reverse proxy เช่น Nginx, Caddy หรือ load balancer ที่ดูแล HTTPS
4. ตั้ง health check ที่ `GET /` และตรวจ log ของ container เป็นระยะ

ตัวอย่างให้ container ทำงานเบื้องหลังและ restart เมื่อเครื่อง reboot:

```bash
docker run -d \
  --restart unless-stopped \
  -p 5000:5000 \
  --name thai-license-plate-ocr \
  thai-license-plate-ocr:latest
```

## ข้อควรระวังสำหรับ production

- ใช้ HTTPS ผ่าน reverse proxy; ไม่ควรเปิด Flask/Gunicorn ตรงสู่ internet
- แอปจำกัดไฟล์อัปโหลดที่ 10 MB แต่ควรตั้ง request-body limit ที่ reverse proxy เพิ่มด้วย
- แอปเป็น stateless: ไม่มี session หรือ cookie และภาพถูกเก็บใน request/page เท่านั้น ไม่มีฐานข้อมูลหรือไฟล์อัปโหลดค้างอยู่ใน container
