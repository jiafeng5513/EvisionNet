import os
import shutil
import zipfile
# 首先引入需要的工具包
# shutil为后期移动文件所需，可以忽略此项
# 源路径
parent_path = r'G:/KittiRaw_zip/2011_10_03'
# 目标路径
target_path = r'H:/KittiRaw/2011_10_03'
# 文件类型选择
# 可以自行更改压缩文件类型，需要引入其它工具包，如tarfile等
# 这里是因为在自己的windows上，zip比较常见，其他类型请自行更改
file_flag = '.zip'
# 删除已解压的zip文件
# 不建议初次使用，在确定程序无误后可以添加使用
def del_old_zip(file_path):
    os.remove(file_path)

# 解压
def decompress(file_path, root):
    # 开始
    # zipfile打开zip文件
    z = zipfile.ZipFile(f'{file_path}', 'r')
    # 解压
    z.extractall(path=f"{root}")    # path为解压路径，解包后位于该路径下
    # 判断是否需要重复解包
    for names in z.namelist():
        if names.endswith(file_flag):
            z.close()
            return 1
    # 结束
    z.close()
    return 0
# 因为我在使用过程中发现有些zip解包后会混在一起
# 在平时大家手动解压时可能也会遇到提示是否覆盖的问题
# 下面的两个函数解决这一问题
# 开始要先创建一个大文件夹 与压缩包名字相同
# 避免后期混乱和麻烦
def start_dir_make(root, dirname):
    os.chdir(root)
    os.mkdir(dirname)
    return os.path.join(root, dirname)

# 去除多余文件夹
def rem_dir_extra(root, father_dir_name):
    # 递归要注意信息的正常处理 搞不好上一个调用已经改变了东西 而下面的调用还是使用之前的数据
    try:
        # 判断文件夹重名 开始
        for item in os.listdir(os.path.join(root, father_dir_name)):
            # 第一步判断是不是一个文件夹，如果不是则跳过本次循环
            if not os.path.isdir(os.path.join(root, father_dir_name, item)):
                continue
            # 判断是否要脱掉一层目录结构
            # 文件夹名字要相同，且子目录中只有单独的一个文件夹
            if item == father_dir_name and len(os.listdir(os.path.join(root, father_dir_name))) == 1:
                # 改变工作目录
                os.chdir(root)
                # 将无用文件夹重命名，因为直接移动会有重名错误
                os.rename(father_dir_name, father_dir_name + '-old')
                # 移动文件后删除空文件夹
                shutil.move(os.path.join(root, father_dir_name + '-old', item), os.path.join(root))
                os.rmdir(os.path.join(root, father_dir_name + '-old'))
                # 将去掉一层目录结构后的文件夹继续作为父本递归处理下去
                # 这里要注意，上面已经发生过数据的改动，所以下面递归传参一定要正确！
                rem_dir_extra(root, item)
            else:
                # 处理那些不满足上面条件的文件夹
                rem_dir_extra(os.path.join(root, father_dir_name), item)
    except Exception as e:
        # 打印错误信息
        print("清除文件夹出错" + str(e))
# 入口

def unzipWithzipfile():
    flag = 1
    while flag:
        # 循环遍历文件夹
        for root, dirs, files in os.walk(parent_path):
            # 读取文件名
            nums = len(files)
            current = 1
            for name in files:
                if name.endswith(file_flag):
                    # 创建文件夹
                    new_ws = start_dir_make(target_path, name.replace(file_flag, ''))
                    # zip文件地址
                    zip_path = os.path.join(root, name)
                    # 解压
                    print("[{}/{}]  unzip:{}...".format(current, nums, f'{root}\\{name}'))
                    flag = decompress(zip_path, new_ws)
                    # 删除解压后的文件
                    # 有点危险
                    # 但不删除又可能会重复运行
                    # 一定要备份或先测试，不然可能会凉，自己选择修改
                    # del_old_zip(zip_path)
                    # 去掉多余的文件结构
                    rem_dir_extra(target_path, name.replace(file_flag, ''))
                    print("{} finished!".format(f'{name}'))
                    current += 1
    # 由于解压可能解了好几次 所以可能会有已经解压好的父级目录重名无法处理 这里要再处理一次
    rem_dir_extra(os.path.split(parent_path)[0], os.path.split(parent_path)[1])
    print("全部解压完成!\n")


if __name__ == '__main__':
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for root, dirs, files in os.walk(parent_path):
        # 读取文件名
        nums = len(files)
        current = 1
        for name in files:
            if name.endswith(file_flag):
                tmp_name = name
                if not os.path.exists(os.path.join(target_path,tmp_name.replace(file_flag, ''))):
                    print("[{}/{}]  unzip:{}...".format(current, nums, f'{root}\\{name}'))
                    zip_path = os.path.join(root, name)
                    cmd = "Bandizip.exe x -y -o:{} {}".format(target_path, zip_path)
                    os.system(cmd)
                    print("{} finished!".format(f'{name}'))
                else:
                    print("[{}/{}]  {} is already unziped!".format(current, nums, f'{name}'))
            else:
                print("[{}/{}]  {} is not a zip file!".format(current, nums, f'{name}'))
            current += 1
    print("全部解压完成!\n")