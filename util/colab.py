import sys
import torch
import requests
import subprocess

# We will need pytorch3d. At the moment (2022-07-23) it's a bit complicated to install in colab:
def install_pytorch_3d_from_prebuilt_wheel_if_needed():
    """
    Attempts to install PyTorch3D – first from FB AI's wheels,
    if that isn't available, fall back to MIT SRG's build.
    """
    need_pytorch3d = False
    try:
        import pytorch3d
        print(f"Pytorch3D was already installed, version: {pytorch3d.__version__}")
    except ModuleNotFoundError:
        print(f"Pytorch3D could not be imported, attempting to install...")
        need_pytorch3d = True
    if not need_pytorch3d:
        return

    # we construct a version string that encodes:
    # Python version, torch's CUDA version, and torch version
    pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
    version_str="".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".",""),
        f"_pyt{pyt_version_str}"
    ])
    print(f"Version string: {version_str}")

    # FB has a magic URL where they offer pre-built wheels
    fbai_wheel_url = f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"

    # We (MIT's SRG) also have our own magic filename and URL as fallback
    srg_filename = f"mit-srg-6s980-colab-wheels-{version_str}.tar.gz"
    srg_wheel_url = f"http://eu.schubert.io/{srg_filename}"

    fb_has_wheel = requests.head(fbai_wheel_url).ok
    srg_has_wheel = requests.head(srg_wheel_url).ok

    if fb_has_wheel or srg_has_wheel:
        print("Found a wheel. First, install some pytorch3d dependencies (fvcore, iopath) that aren't included in pre-built wheel:")
        subprocess.call([sys.executable, '-m', 'pip', 'install', '--quiet', 'fvcore', 'iopath'])
#        ! pip install --quiet fvcore iopath  # type: ignore

    if fb_has_wheel:
        print("Found FB AI wheel, installing…")
        subprocess.call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-cache-dir', 'pytorch3d', '-f', fbai_wheel_url])
#        ! pip install --no-index --no-cache-dir pytorch3d -f {fbai_wheel_url}  # type: ignore
    elif srg_has_wheel:
        print("Found only SRG wheel, installing...")
        subprocess.call(f'curl -L {srg_wheel_url} | tar xz'.split())
        subprocess.call(f'{sys.executable} -m pip install --no-index --find-links=./wheeldir pytorch3d'.split())
#        ! curl -L {srg_wheel_url} | tar xz  # type: ignore
#        ! pip install --no-index --find-links=./wheeldir pytorch3d  # type: ignore
    else:
        raise RuntimeError("Can't find any pre-compiled pytorch3D wheel. :/")
