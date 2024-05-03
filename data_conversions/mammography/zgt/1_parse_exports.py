import json
import pandas as pd
import os

export_root = 'data/mammography/zgt/label_studio/veltman'


def get_draft_authors(row):
    if row.get('drafts') is not None:
        return [draft['user'] for draft in row['drafts']]
    return None


def get_regions(row):
    ls_annotation = row['annotations'][0]
    common = dict(id=row['id'], image_path=row['data']['image'].split('/data/local-files/?d=')[1])  # image level info
    common['image_name'] = common['image_path'].split('/')[-1]
    for r in ls_annotation['result']:
        if r.get('type') == 'choices':
            common[r['to_name']] = r['value']['choices'][0]
        if r.get('type') == 'brushlabels':
            region_key = r['value']['brushlabels'][0]
            assert common.get(region_key) is None
            region_info = dict(rle=r['value']['rle'], height=r['original_height'], width=r['original_width'])
            common[region_key] = region_info
    # loop through the regions
    regions = []
    for region_key in [f'R{i}' for i in range(1, 6)]:
        if common.get(region_key) is None:
            continue
        region = common[region_key]
        del common[region_key]
        for k in list(common.keys()):
            if not k.endswith(region_key):
                continue
            region[k[:-len(region_key)]] = common[k]
            del common[k]
        regions.append(region)
    common['regions'] = regions

    # check for any remaining drafts
    common['draft_authors'] = get_draft_authors(row)
    return common


def stub_missing_rle(row):
    rle = row['rle']
    if isinstance(rle, list):
        return rle
    return []


def main():
    with open(os.path.join(export_root, 'project-4-at-2024-04-29-10-27-5227907f.json')) as f:
        data1 = json.load(f)
    with open(os.path.join(export_root, 'project-8-at-2024-04-29-10-27-483fed16.json')) as f:
        data2 = json.load(f)

    data = data1 + data2
    df = pd.DataFrame([get_regions(row) for row in data])
    df = df.explode('regions', ignore_index=True).rename(columns={'regions': 'region'})
    regions = df['region'].apply(pd.Series).drop(columns=[0])
    df = pd.concat([df.drop(columns=['region']), regions], axis=1)
    df['rle'] = df.apply(stub_missing_rle, axis=1)

    with open(os.path.join(export_root, 'parsed_exports.csv'), 'w') as f:
        df.drop(columns=['draft_authors']).to_csv(f, index=False)
    files = list(df['image_path'].unique())
    with open(os.path.join(export_root, 'annotated_images.txt'),  'w') as f:
        f.writelines('\n'.join(files))


if __name__ == '__main__':
    main()
