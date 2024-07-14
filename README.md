# Kho nhạc và lời tách ra từ Youtube music videos

## Động lực

1. Nhu cầu hát với beat
2. Luyện nhạc cụ với vocal sẽ hiệu quả hơn

Các dịch vụ tách beat và vocal hiện có khá nhiều, tuy nhiên:
- Chỉ có thể xử lý 1 audio tại 1 thời điểm
- Cần khá nhiều bước để thực hiện: Lấy URL -> tải video -> upload để xử lý -> tải audio về
- Mất 2 - 10 phút để xử lý

## Thông tin thư mục

- Thư mục `beat`: chứa phần nhạc của video
- Thư mục `vocal`: chứa phần lời của video
- File `list.xlsx`: danh sách thông tin các video 

## Hướng dẫn tra cứu

Làm lần lượt theo các bước sau:
1. Mở file `list.xlsx`, tra cứu theo `title` hoặc `url` để lấy `id`
2. Vào thư mục `beats` hoặc `vocals` tùy mục đích cá nhân
3. Tìm file mp3 theo `id` bên trên 

> Nếu không tìm thấy file audio mình cần, hãy đề xuất bổ sung theo hướng dẫn bên dưới

## Hướng dẫn đề xuất

Làm lần lượt theo các bước sau:
1. Lấy URL các video bạn cần
2. Sắp xếp các URL theo từng hàng như sau:

	```
	URL1
	URL2
	...
	```

3. Gửi trực tiếp danh sách trên đến Messenger [này](https://www.facebook.com/a2lucifer)

> Nếu chưa biết cách lấy URL video trên smartphone, bạn có thể làm theo hướng dẫn sau:
- [Hướng dẫn lấy URL video Youtube trên Android](https://www.wikihow.com/Copy-a-URL-on-the-YouTube-App-on-Android)
- [Hướng dẫn lấy URL video Youtube trên iOS](https://www.wikihow.com/Copy-a-URL-on-the-YouTube-App-on-iPhone-or-iPad)

## Báo cáo chất lượng audio

Nếu phát hiện file audio nào có chất lượng kém, hãy phản hồi vào [đây](https://www.facebook.com/a2lucifer)

## Ưu điểm

1. Được lưu trữ tập trung
2. Lấy nhiều audio nhanh chóng 
## Nhược điểm

1. Phản hồi và bổ sung audio theo yêu cầu chậm do thu thập yêu cầu thủ công
2. Chưa tách được phần bass và drums
3. Người dùng vẫn phải tìm kiếm thủ công

## Lời cảm ơn
Chân thành cảm ơn các dự án opensource sau:
- [Vocal Remover](https://github.com/tsurumeso/vocal-remover)
- [Rclone](https://github.com/rclone/rclone)
