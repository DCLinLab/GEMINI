from pathlib import Path
from cli import VesselCrystalCLI


if __name__ == '__main__':
    vs = VesselCrystalCLI()
    imgs = [
        r"Y:\All Data\Lab Members\Jiaxi Lu\01_Raw Data\Tickertape\2025-26\Cancer_GEAR\InVivo\20250324_NudeMice26-33_LPSDoseDependency_Snap_LSM780\Nude32_CD31-IHC\10x_Scan\Z3\Nude32_Z3_2025_05_29__19_01_45.czi",
        r"Y:\All Data\Lab Members\Jiaxi Lu\01_Raw Data\Tickertape\2025-26\Cancer_GEAR\InVivo\20250324_NudeMice26-33_LPSDoseDependency_Snap_LSM780\Nude32_CD31-IHC\10x_Scan\Z5\Nude32_Z5_2025_05_29__20_55_14.czi",
        r"Y:\All Data\Lab Members\Jiaxi Lu\01_Raw Data\Tickertape\2025-26\Cancer_GEAR\InVivo\20250324_NudeMice26-33_LPSDoseDependency_Snap_LSM780\Nude32_CD31-IHC\10x_Scan\Z6\Nude32_Z6_2025_06_01__19_37_35.czi",
        r"Y:\All Data\Lab Members\Jiaxi Lu\01_Raw Data\Tickertape\2025-26\Cancer_GEAR\InVivo\20250324_NudeMice26-33_LPSDoseDependency_Snap_LSM780\Nude32_CD31-IHC\10x_Scan\Z8\Nude32_Z8_2025_06_02__18_05_01.czi",
        r"Y:\All Data\Lab Members\Jiaxi Lu\01_Raw Data\Tickertape\2025-26\Cancer_GEAR\InVivo\20250324_NudeMice26-33_LPSDoseDependency_Snap_LSM780\Nude32_CD31-IHC\10x_Scan\Z9\Nude32_Z9_2025_05_30__00_49_24.czi"
    ]
    nworkers = 10
    outdir = Path(r'D:\Zuohan\vessel_batch')
    for i in imgs:
        p = Path(i)
        outpath = outdir / p.parts[-2] / 'crystal'
        # vs.segment_crystal_by_tile(i, str(outpath), workers=nworkers)
        vs.ratiometrics(str(outpath), str(p.parent), workers=nworkers)
        # vs.segment_vessel(i, str(outpath.parent), workers=nworkers)
        # vs.assemble_crystal(str(outpath), str(p.parent), workers=nworkers)
