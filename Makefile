train-x86-windows-and-extract:
	python train.py --targets="curl, openssl, httpd, sqlite3, libcrypto"
	python extract.py --targets="curl, openssl, httpd, sqlite3, libcrypto"

train-x86-windows-and-extract-50:
	python train.py --targets="curl, openssl, httpd, sqlite3, libcrypto" --proportion 5
	python extract.py --targets="curl, openssl, httpd, sqlite3, libcrypto" --proportion 5

train-x86-aarch64-and-extract:
	python train.py --targets="libcrypto-xarch, libc"
	python extract.py --targets="libcrypto-xarch, libc"

train-x86-aarch64-and-extract-50:
	python train.py --targets="libcrypto-xarch, libc" --proportion 5
	python extract.py --targets="libcrypto-xarch, libc" --proportion 5

train-independently-and-extract:
	python train.py --targets="libc"
	python extract.py --targets="libc"
	python train.py --targets="curl"
	python extract.py --targets="curl"
	python train.py --targets="openssl"
	python extract.py --targets="openssl"
	python train.py --targets="httpd"
	python extract.py --targets="httpd"
	python train.py --targets="sqlite3"
	python extract.py --targets="sqlite3"
	python train.py --targets="libcrypto"
	python extract.py --targets="libcrypto"
	python train.py --targets="libcrypto-xarch"
	python extract.py --targets="libcrypto-xarch"
validation:
	python validation.py