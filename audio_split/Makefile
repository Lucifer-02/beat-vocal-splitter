clean:
	rm audio/*
	rm split/music/*
	rm split/vocal/*

run:
	python3 auto.py && rclone sync split/ lucifer_drive:split --verbose

upload:
	rclone sync split/ lucifer_drive:split --verbose
