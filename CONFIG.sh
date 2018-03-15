# set rsa
mkdir ~/.ssh 
cd ~/.ssh
qrsctl get northrend kirk_rsa.tar.gz ~/.ssh/kirk_rsa.tar.gz
tar -xzvf kirk_rsa.tar.gz
rm kirk_rsa.tar.gz

# set git
cd -
git config --global user.email "286424061@qq.com"
git config --global user.name "Northrend"
git remote remove origin
git remote add origin git@github.com:Northrend/Kaels-toolbox.git

