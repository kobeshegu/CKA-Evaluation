import os
import random
import shutil
import zipfile
import click

def chooseFiledir(filedir,outdir,number):
    res=0
    os.makedirs(outdir, exist_ok=True)
    pathDir = os.listdir(filedir)
    sample = random.sample(pathDir, number)
    for name in sample:
        print(name)
        first_name=os.path.join(filedir, name)
        out_name=os.path.join(outdir, name)
        shutil.copy(first_name,out_name)
        res+=1
    return res

def chooseFilezip(filezip,OutDir,number):
    res=0
    os.makedirs(OutDir, exist_ok=True)
    with zipfile.ZipFile(filezip,'r') as zfile:
        nl=zfile.namelist()
        il=zfile.infolist()
        picknumber=int(number)
        namelist=[n for n in nl if 'jpg' in n or 'png' in n or 'jpeg' in n]
        sample = random.sample(namelist, picknumber)
        for name in sample:
            if 'jpg' in name or 'png' in name or 'jpeg' in name:
                res+=1
                print(name)
                zfile.extract(name,path=OutDir)
        zfile.close()
    return res

def rm_file(path, maindir):
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)):
            rm_file(os.path.join(path, i),maindir)
            if os.path.exists(os.path.join(path,i)):
                os.removedirs(os.path.join(path,i))
        elif os.path.isfile(os.path.join(path,i)):
            if not os.path.exists(os.path.join(maindir,i)):
                shutil.move(os.path.join(path, i),os.path.join(maindir,i))
            else:
                if path != maindir:
                    os.remove(os.path.join(path,i))

@click.command()
@click.option('--real_dataset', help='Full real dataset', type=str, default=None, metavar='[ZIP|DIR]', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--number', help='Number of the chosen subset', type=int, default=50000, metavar='INT', show_default=True)
def choose_subset(
    real_dataset: str,
    outdir: str,
    number: int,
):

    filezip = real_dataset
    filedir = real_dataset
    outdir = outdir
    number=number
    #os.makedirs(outdir, exist_ok=True)
    if os.path.isdir(real_dataset):
        res = chooseFiledir(filedir,outdir,number)
        print(res)
        rm_file(outdir,outdir)
    elif zipfile.is_zipfile(real_dataset):
        res = chooseFilezip(filezip,outdir,number)
        print(res)
        rm_file(outdir,outdir)
    else:
        raise ValueError

if __name__ == '__main__':
    choose_subset()
