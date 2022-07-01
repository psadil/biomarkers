import argparse
import os
from pathlib import Path

from nipype import Function
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from ..interfaces.first import FIRST
# from nipype.interfaces.fsl import FIRST

# from niworkflows.interfaces import bids

#TODO: generate report to check registration

def get_volume(segmentation_file: str, label: int) -> dict:
    import nibabel as nib
    import numpy as np
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide#Labels
    labels = {
        'Left-Thalamus-Proper': 10,
        'Left-Caudate': 11,
        'Left-Putamen': 12,
        'Left-Pallidum': 13,
        'Brain-Stem /4th Ventricle': 16,
        'Left-Hippocampus': 17,
        'Left-Amygdala': 18,
        'Left-Accumbens-area': 26,
        'Right-Thalamus-Proper': 49,
        'Right-Caudate': 50,
        'Right-Putamen': 51,
        'Right-Pallidum': 52,
        'Right-Hippocampus': 53,
        'Right-Amygdala': 54,
        'Right-Accumbens-area': 58 
    }
    if label not in labels.keys():
        raise ValueError(f'label {label} not in known list -- {labels.keys()}')
    nii = nib.load(segmentation_file).get_fdata()
    return {label: np.sum(nii==labels.get(label), dtype=int)}


def get_basename(in_file: str) -> str:
    import os
    return os.path.basename(in_file)


def join_volumes(volumes: list[dict], src: str, path: str = "volumes.json") -> str:
    import os
    import pandas as pd
    vol_dict = {}
    for vol in volumes:
        vol_dict.update(vol)
    out = pd.DataFrame(vol_dict, index=[os.path.basename(src)])
    out['method'] = 'FIRST'
    f = os.path.abspath(path)
    # out.to_csv(f, index_label='source')
    out.to_json(f, orient="index")
    return f


def init_first_wf() -> pe.Workflow:
    inputnode = pe.Node(
        niu.IdentityInterface(
          fields=['in_file']), 
          name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
          fields=['bvars', 'original_segmentations', 'segmentation_file', 'vtk_surfaces', 'volume_table', "to_std_sub_nii", "to_std_sub_mat"]), 
          name='outputnode')  
              
  # FIRST - Image Segmentation
    first = pe.Node(
      FIRST(),
      name="segmentation"
    )
 
    basename = pe.Node(
        Function(
            input_names=["in_file"],
            output_names=["basename"],
            function=get_basename
        ),
        name="get_basename"
    )

    volumes = pe.Node(
        Function(
            input_names=["segmentation_file", "label"],
            output_names=["volume"],
            function=get_volume),
        name='get_volume')
    volumes.iterables = [(
        'label', 
        [
            'Left-Amygdala', 'Right-Amygdala', 
            'Left-Hippocampus', 'Right-Hippocampus'
        ]
     )] 
    volume_table = pe.JoinNode(
        Function(
            function=join_volumes,
            input_names=['volumes', 'src'],
            output_names=['volume_table']),
        name='join_volumes',
        joinsource=volumes,
        joinfield=['volumes']
     )

    wf = pe.Workflow(name=f'first_wf')

    wf.connect([
        (inputnode, first, [('in_file', 'in_file')]),
        (inputnode, basename, [('in_file', 'in_file')]),
        (basename, first, [('basename', 'out_file')]),
        (first, volumes, [('segmentation_file', 'segmentation_file')]),
        (first, outputnode, [
            ('bvars', 'bvars'),
            ('segmentation_file', 'segmentation_file'),
            ('vtk_surfaces', 'vtk_surfaces'),
            ("to_std_sub_nii", "to_std_sub_nii"),
            ("to_std_sub_mat", "to_std_sub_mat")
            ]),
        (volumes, volume_table, [('volume', 'volumes')]),
        (inputnode, volume_table, [('in_file', 'src')]),
        (volume_table, outputnode, [('volume_table', 'volume_table')])
         ])

    return wf


def init_anat_wf(
    img: list[Path],
    output_dir: str = "output",
    work_dir: str = "work") -> None:

    inputnode = pe.Node(
        niu.IdentityInterface(
          fields=['in_file']), 
        name='inputnode')  
    inputnode.iterables = [('in_file', img)]

    datasink = pe.Node(
        nio.DataSink(
            base_directory=output_dir,
            regexp_substitutions=[
                (r'_in_file_.*/', ''),
                # (r'\.nii\.gz\\', ''),
            ]
        ), 
        name='sinker',
        run_without_submitting=True)

    wf = pe.Workflow(name="anatomical", base_dir=work_dir)
    first = init_first_wf()
    wf.connect([
        (inputnode, first, [('in_file', 'inputnode.in_file')]),
        (first, datasink, [
            ('outputnode.bvars', '@bvars'),
            ('outputnode.original_segmentations', '@original_segmentations'),
            ('outputnode.segmentation_file', '@segmentation_file'),
            ('outputnode.vtk_surfaces', '@vtk_surfaces'),
            ('outputnode.volume_table', '@volume_table'),
            ('outputnode.to_std_sub_nii', '@to_std_sub_nii'),
            ('outputnode.to_std_sub_mat', '@to_std_sub_mat'),
            ]) 
        ])

    return wf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "img",  
        nargs="+", 
        help="T1w images to process")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work",
        help="path where intermediate results will be stored and cached")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out",
        help="directory in which results will be stored")
    parser.add_argument(
        "--plugin",
        type=str,
        default="Linear",
        help="nipype plugin name")

    args = parser.parse_args()
    anat_wf = init_anat_wf(
        img=[Path(x).absolute() for x in args.img],
        output_dir=os.path.abspath(args.out_dir),
        work_dir=os.path.abspath(args.work_dir))
    
    anat_wf.run(args.plugin)


if __name__ == "__main__":
    main()
