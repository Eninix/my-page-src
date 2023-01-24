echo '==> Prepare to Push'
git add .
git commit -m 'auto update by script'

echo '==> Start Push'
git push

echo '==> Push Complete'
echo ''
echo 'page-src Done.'
