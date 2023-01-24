SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
cd $SCRIPT_DIR

echo '=======>hexo deploy'
hexo clean && hexo g -D
echo ''

echo '=======>push to github'
./push-to-github.sh


# # 使用 auto.sh 1 来push子模块
# if [ $# -ge 1 ] ;then
# 	if [ $1 -eq 1 ] ;then
# 		cd themes/next
# 		echo ''
# 		echo ''
# 		./push-to-github.sh
# 	fi
# fi


