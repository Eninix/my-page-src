---
title: 关于Archlinux无法保存密码
abbrlink: 1ccad7af
date: 2022-11-12 23:47:44
tags: ['linux','archlinux']
categories: Archlinux
---

在使用Arch的过程中，我发现vscode的登陆同步每次打开都需要重新进行登陆。关于这一点，经过网上搜集资料，发现是由于缺少gnome-keyring导致。

除了vscode，我的世界官方启动器以及其他的一些软件也需要gnome-keyring.

解决方法如下
```shell
sudo pacman -S gnome-keyring
# 如果需要使用gui管理密码可以安装 seahorse
# sudo pacman -S seahorse
```
