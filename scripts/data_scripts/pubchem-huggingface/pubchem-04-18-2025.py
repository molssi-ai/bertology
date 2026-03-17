# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""loading script for the pubchem-04-18-2025 data"""

import ijson
import datasets


_CITATION = """\
@article{Kim:2022:D1373,
    author = {Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and Zaslavsky, Leonid and Zhang, Jian and Bolton, Evan E},
    title = "{PubChem 2023 update}",
    journal = {Nucleic Acids Research},
    volume = {51},
    pages = {D1373-D1380},
    year = {2022},
    doi = {10.1093/nar/gkac956}
}
"""

_DESCRIPTION = """\
PubChem (https://pubchem.ncbi.nlm.nih.gov) is a popular chemical information \
resource that serves a wide range of use cases. In the past \
two years, a number of changes were made to PubChem. Data from more than 120 \
data sources was added to PubChem. Some major highlights include: the \
integration of Google Patents data into PubChem, which greatly expanded the \
coverage of the PubChem Patent data collection; the creation of the Cell Line \
and Taxonomy data collections, which provide quick and easy access to chemical \
information for a given cell line and taxon, respectively; and the update of \
the bioassay data model. In addition, new functionalities were added to the \
PubChem programmatic access protocols, PUG-REST and PUG-View, including support \
for target-centric data download for a given protein, gene, pathway, cell line, \
and taxon and the addition of the `standardize` option to PUG-REST, which \
returns the standardized form of an input chemical structure. A significant \
update was also made to PubChemRDF. The present paper provides an overview of \
these changes.
"""

_HOMEPAGE = "https://pubchem.ncbi.nlm.nih.gov"

_LICENSE = "https://www.ncbi.nlm.nih.gov/home/about/policies/#data"

_BASE_DATA_URL_FORMAT_STR = "https://huggingface.co/datasets/molssiai-hub/pubchem-04-18-2025/resolve/main/data/train/Compound_{range}.json"

_CONFIG_NAMES = [
    "pubchem-04-18-2025"
]

_FEATURE_KEYS = [
    'PUBCHEM_COMPOUND_CID',
    'PUBCHEM_COMPOUND_CANONICALIZED',
    'PUBCHEM_CACTVS_COMPLEXITY',
    'PUBCHEM_CACTVS_HBOND_ACCEPTOR',
    'PUBCHEM_CACTVS_HBOND_DONOR',
    'PUBCHEM_CACTVS_ROTATABLE_BOND',
    'PUBCHEM_CACTVS_SUBSKEYS',
    'PUBCHEM_IUPAC_OPENEYE_NAME',
    'PUBCHEM_IUPAC_CAS_NAME',
    'PUBCHEM_IUPAC_NAME_MARKUP',
    'PUBCHEM_IUPAC_NAME',
    'PUBCHEM_IUPAC_SYSTEMATIC_NAME',
    'PUBCHEM_IUPAC_TRADITIONAL_NAME',
    'PUBCHEM_IUPAC_INCHI',
    'PUBCHEM_IUPAC_INCHIKEY',
    'PUBCHEM_XLOGP3_AA',
    'PUBCHEM_EXACT_MASS',
    'PUBCHEM_MOLECULAR_FORMULA',
    'PUBCHEM_MOLECULAR_WEIGHT',
    'PUBCHEM_SMILES',
    'PUBCHEM_OPENEYE_CAN_SMILES',
    'PUBCHEM_OPENEYE_ISO_SMILES',
    'PUBCHEM_CACTVS_TPSA',
    'PUBCHEM_MONOISOTOPIC_WEIGHT',
    'PUBCHEM_TOTAL_CHARGE',
    'PUBCHEM_HEAVY_ATOM_COUNT',
    'PUBCHEM_ATOM_DEF_STEREO_COUNT',
    'PUBCHEM_ATOM_UDEF_STEREO_COUNT',
    'PUBCHEM_BOND_DEF_STEREO_COUNT',
    'PUBCHEM_BOND_UDEF_STEREO_COUNT',
    'PUBCHEM_ISOTOPIC_ATOM_COUNT',
    'PUBCHEM_COMPONENT_COUNT',
    'PUBCHEM_CACTVS_TAUTO_COUNT',
    'PUBCHEM_COORDINATE_TYPE',
    'PUBCHEM_BONDANNOTATIONS',
    'COORDS',
    'ATOMIC_INDICES',
    'ATOMIC_SYMBOLS',
    'ATOMIC_NUMBERS',
    'ATOMIC_FORMAL_CHARGES',
    'BOND_ORDERS',
    'PUBCHEM_XLOGP3',
    'PUBCHEM_NONSTANDARDBOND',
    'PUBCHEM_REFERENCE_STANDARDIZATION',
]

_FEATURE_TYPES = datasets.Features(
    {
        'PUBCHEM_COMPOUND_CID': datasets.Value(dtype='string', id=None),
        'PUBCHEM_COMPOUND_CANONICALIZED': datasets.Value(dtype='string', id=None),
        'PUBCHEM_CACTVS_COMPLEXITY': datasets.Value(dtype='float64', id=None),
        'PUBCHEM_CACTVS_HBOND_ACCEPTOR': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_CACTVS_HBOND_DONOR': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_CACTVS_ROTATABLE_BOND': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_CACTVS_SUBSKEYS': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_OPENEYE_NAME': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_CAS_NAME': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_NAME_MARKUP': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_NAME': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_SYSTEMATIC_NAME': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_TRADITIONAL_NAME': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_INCHI': datasets.Value(dtype='string', id=None),
        'PUBCHEM_IUPAC_INCHIKEY': datasets.Value(dtype='string', id=None),
        'PUBCHEM_XLOGP3_AA': datasets.Value(dtype='float64', id=None),
        'PUBCHEM_EXACT_MASS': datasets.Value(dtype='float64', id=None),
        'PUBCHEM_MOLECULAR_FORMULA': datasets.Value(dtype='string', id=None),
        'PUBCHEM_MOLECULAR_WEIGHT': datasets.Value(dtype='float64', id=None),
        'PUBCHEM_SMILES': datasets.Value(dtype='string', id=None),
        'PUBCHEM_OPENEYE_CAN_SMILES': datasets.Value(dtype='string', id=None),
        'PUBCHEM_OPENEYE_ISO_SMILES': datasets.Value(dtype='string', id=None),
        'PUBCHEM_CACTVS_TPSA': datasets.Value(dtype='float64', id=None),
        'PUBCHEM_MONOISOTOPIC_WEIGHT': datasets.Value(dtype='float64', id=None),
        'PUBCHEM_TOTAL_CHARGE': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_HEAVY_ATOM_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_ATOM_DEF_STEREO_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_ATOM_UDEF_STEREO_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_BOND_DEF_STEREO_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_BOND_UDEF_STEREO_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_ISOTOPIC_ATOM_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_COMPONENT_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_CACTVS_TAUTO_COUNT': datasets.Value(dtype='int64', id=None),
        'PUBCHEM_COORDINATE_TYPE': datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None),
        'PUBCHEM_BONDANNOTATIONS': datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None),
        'COORDS': datasets.Sequence(feature=datasets.Value(dtype='float64', id=None), length=-1, id=None),
        'ATOMIC_INDICES': datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None),
        'ATOMIC_SYMBOLS': datasets.Sequence(feature=datasets.Value(dtype='string', id=None), length=-1, id=None),
        'ATOMIC_NUMBERS': datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None),
        'ATOMIC_FORMAL_CHARGES': datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None),
        'BOND_ORDERS': datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None),
        'PUBCHEM_XLOGP3': datasets.Value(dtype='string', id=None),
        'PUBCHEM_NONSTANDARDBOND': datasets.Value(dtype='string', id=None),
        'PUBCHEM_REFERENCE_STANDARDIZATION': datasets.Value(dtype='string', id=None)
    }
)

_ID_RANGES = [
    '000000001_000500000',
    '000500001_001000000',
    '001000001_001500000',
    '001500001_002000000',
    '002000001_002500000',
    '002500001_003000000',
    '003000001_003500000',
    '003500001_004000000',
    '004000001_004500000',
    '004500001_005000000',
    '005000001_005500000',
    '005500001_006000000',
    '006000001_006500000',
    '006500001_007000000',
    '007000001_007500000',
    '007500001_008000000',
    '008000001_008500000',
    '008500001_009000000',
    '009000001_009500000',
    '009500001_010000000',
    '010000001_010500000',
    '010500001_011000000',
    '011000001_011500000',
    '011500001_012000000',
    '012000001_012500000',
    '012500001_013000000',
    '013000001_013500000',
    '013500001_014000000',
    '014000001_014500000',
    '014500001_015000000',
    '015000001_015500000',
    '015500001_016000000',
    '016000001_016500000',
    '016500001_017000000',
    '017000001_017500000',
    '017500001_018000000',
    '018000001_018500000',
    '018500001_019000000',
    '019000001_019500000',
    '019500001_020000000',
    '020000001_020500000',
    '020500001_021000000',
    '021000001_021500000',
    '021500001_022000000',
    '022000001_022500000',
    '022500001_023000000',
    '023000001_023500000',
    '023500001_024000000',
    '024000001_024500000',
    '024500001_025000000',
    '025000001_025500000',
    '025500001_026000000',
    '026000001_026500000',
    '026500001_027000000',
    '027000001_027500000',
    '027500001_028000000',
    '028000001_028500000',
    '028500001_029000000',
    '029000001_029500000',
    '029500001_030000000',
    '030000001_030500000',
    '030500001_031000000',
    '031000001_031500000',
    '031500001_032000000',
    '032000001_032500000',
    '032500001_033000000',
    '033000001_033500000',
    '033500001_034000000',
    '034000001_034500000',
    '034500001_035000000',
    '035000001_035500000',
    '035500001_036000000',
    '036000001_036500000',
    '036500001_037000000',
    '037000001_037500000',
    '037500001_038000000',
    '038000001_038500000',
    '038500001_039000000',
    '039000001_039500000',
    '039500001_040000000',
    '040000001_040500000',
    '040500001_041000000',
    '041000001_041500000',
    '041500001_042000000',
    '042000001_042500000',
    '042500001_043000000',
    '043000001_043500000',
    '043500001_044000000',
    '044000001_044500000',
    '044500001_045000000',
    '045000001_045500000',
    '045500001_046000000',
    '046000001_046500000',
    '046500001_047000000',
    '047000001_047500000',
    '047500001_048000000',
    '048000001_048500000',
    '048500001_049000000',
    '049000001_049500000',
    '049500001_050000000',
    '050000001_050500000',
    '050500001_051000000',
    '051000001_051500000',
    '051500001_052000000',
    '052000001_052500000',
    '052500001_053000000',
    '053000001_053500000',
    '053500001_054000000',
    '054000001_054500000',
    '054500001_055000000',
    '055000001_055500000',
    '055500001_056000000',
    '056000001_056500000',
    '056500001_057000000',
    '057000001_057500000',
    '057500001_058000000',
    '058000001_058500000',
    '058500001_059000000',
    '059000001_059500000',
    '059500001_060000000',
    '060000001_060500000',
    '060500001_061000000',
    '061000001_061500000',
    '061500001_062000000',
    '062000001_062500000',
    '062500001_063000000',
    '063000001_063500000',
    '063500001_064000000',
    '064000001_064500000',
    '064500001_065000000',
    '065000001_065500000',
    '065500001_066000000',
    '066000001_066500000',
    '066500001_067000000',
    '067000001_067500000',
    '067500001_068000000',
    '068000001_068500000',
    '068500001_069000000',
    '069000001_069500000',
    '069500001_070000000',
    '070000001_070500000',
    '070500001_071000000',
    '071000001_071500000',
    '071500001_072000000',
    '072000001_072500000',
    '072500001_073000000',
    '073000001_073500000',
    '073500001_074000000',
    '074000001_074500000',
    '074500001_075000000',
    '075000001_075500000',
    '075500001_076000000',
    '076000001_076500000',
    '076500001_077000000',
    '077000001_077500000',
    '077500001_078000000',
    '078000001_078500000',
    '078500001_079000000',
    '079000001_079500000',
    '079500001_080000000',
    '080000001_080500000',
    '080500001_081000000',
    '081000001_081500000',
    '081500001_082000000',
    '082000001_082500000',
    '082500001_083000000',
    '083000001_083500000',
    '083500001_084000000',
    '084000001_084500000',
    '084500001_085000000',
    '085000001_085500000',
    '085500001_086000000',
    '086000001_086500000',
    '086500001_087000000',
    '087000001_087500000',
    '087500001_088000000',
    '088000001_088500000',
    '088500001_089000000',
    '089000001_089500000',
    '089500001_090000000',
    '090000001_090500000',
    '090500001_091000000',
    '091000001_091500000',
    '091500001_092000000',
    '092000001_092500000',
    '092500001_093000000',
    '093000001_093500000',
    '093500001_094000000',
    '094000001_094500000',
    '094500001_095000000',
    '095000001_095500000',
    '095500001_096000000',
    '096000001_096500000',
    '096500001_097000000',
    '097000001_097500000',
    '097500001_098000000',
    '098000001_098500000',
    '098500001_099000000',
    '099000001_099500000',
    '099500001_100000000',
    '100000001_100500000',
    '100500001_101000000',
    '101000001_101500000',
    '101500001_102000000',
    '102000001_102500000',
    '102500001_103000000',
    '103000001_103500000',
    '103500001_104000000',
    '104000001_104500000',
    '104500001_105000000',
    '105000001_105500000',
    '105500001_106000000',
    '106000001_106500000',
    '106500001_107000000',
    '107000001_107500000',
    '107500001_108000000',
    '108000001_108500000',
    '108500001_109000000',
    '109000001_109500000',
    '109500001_110000000',
    '110000001_110500000',
    '110500001_111000000',
    '111000001_111500000',
    '111500001_112000000',
    '112000001_112500000',
    '112500001_113000000',
    '113000001_113500000',
    '113500001_114000000',
    '114000001_114500000',
    '114500001_115000000',
    '115000001_115500000',
    '115500001_116000000',
    '116000001_116500000',
    '116500001_117000000',
    '117000001_117500000',
    '117500001_118000000',
    '118000001_118500000',
    '118500001_119000000',
    '119000001_119500000',
    '119500001_120000000',
    '120000001_120500000',
    '120500001_121000000',
    '121000001_121500000',
    '121500001_122000000',
    '122000001_122500000',
    '122500001_123000000',
    '123000001_123500000',
    '123500001_124000000',
    '124000001_124500000',
    '124500001_125000000',
    '125000001_125500000',
    '125500001_126000000',
    '126000001_126500000',
    '126500001_127000000',
    '127000001_127500000',
    '127500001_128000000',
    '128000001_128500000',
    '128500001_129000000',
    '129000001_129500000',
    '129500001_130000000',
    '130000001_130500000',
    '130500001_131000000',
    '131000001_131500000',
    '131500001_132000000',
    '132000001_132500000',
    '132500001_133000000',
    '133000001_133500000',
    '133500001_134000000',
    '134000001_134500000',
    '134500001_135000000',
    '135000001_135500000',
    '135500001_136000000',
    '136000001_136500000',
    '136500001_137000000',
    '137000001_137500000',
    '137500001_138000000',
    '138000001_138500000',
    '138500001_139000000',
    '139000001_139500000',
    '139500001_140000000',
    '140000001_140500000',
    '140500001_141000000',
    '141000001_141500000',
    '141500001_142000000',
    '142000001_142500000',
    '142500001_143000000',
    '143000001_143500000',
    '143500001_144000000',
    '144000001_144500000',
    '144500001_145000000',
    '145000001_145500000',
    '145500001_146000000',
    '146000001_146500000',
    '146500001_147000000',
    '147000001_147500000',
    '147500001_148000000',
    '148000001_148500000',
    '148500001_149000000',
    '149000001_149500000',
    '149500001_150000000',
    '150000001_150500000',
    '150500001_151000000',
    '151000001_151500000',
    '151500001_152000000',
    '152000001_152500000',
    '152500001_153000000',
    '153000001_153500000',
    '153500001_154000000',
    '154000001_154500000',
    '154500001_155000000',
    '155000001_155500000',
    '155500001_156000000',
    '156000001_156500000',
    '156500001_157000000',
    '157000001_157500000',
    '157500001_158000000',
    '158000001_158500000',
    '158500001_159000000',
    '159000001_159500000',
    '159500001_160000000',
    '160000001_160500000',
    '160500001_161000000',
    '161000001_161500000',
    '161500001_162000000',
    '162000001_162500000',
    '162500001_163000000',
    '163000001_163500000',
    '163500001_164000000',
    '164000001_164500000',
    '164500001_165000000',
    '165000001_165500000',
    '165500001_166000000',
    '166000001_166500000',
    '166500001_167000000',
    '167000001_167500000',
    '167500001_168000000',
    '168000001_168500000',
    '168500001_169000000',
    '169000001_169500000',
    '169500001_170000000',
    '170000001_170500000',
    '170500001_171000000',
    '171000001_171500000',
    '171500001_172000000',
    '172000001_172500000',
    '172200001_173000000'
]


class PubChemConfig(datasets.BuilderConfig):
    """
    Configuration class for the PubChem dataset.
    Args:
        features (dict): A dictionary specifying the features of the dataset.
        data_url (str): The URL to download the dataset.
        url (str): The URL to the dataset homepage.
        citation (str): The citation for the dataset.
        **kwargs: Additional keyword arguments.
    Attributes:
        version (datasets.Version): The version of the dataset.
    """

    def __init__(self, features, url, citation, id_range, **kwargs):
        super(PubChemConfig, self).__init__(
            version=datasets.Version("0.0.0"), **kwargs
        )
        self.features = features
        self.url = url
        self.citation = citation
        self.id_range = id_range


class PubChem(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = PubChemConfig

    DEFAULT_CONFIG_NAME = "pubchem-04-18-2025"

    VERSION = datasets.Version("0.0.0")

    url = "https://huggingface.co/datasets/molssiai-hub/pubchem-04-18-2025"

    BUILDER_CONFIGS = [
        PubChemConfig(
            name="pubchem-04-18-2025",
            description=_DESCRIPTION,
            features=_FEATURE_KEYS,
            url=url,
            citation=_CITATION,
            id_range=_ID_RANGES,
        )
    ]

    def _info(self):
        if self.config.name in _CONFIG_NAMES:
            features = _FEATURE_TYPES
        else:
            raise NotImplementedError("Unknown configuration in _info")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            supervised_keys=None,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name in _CONFIG_NAMES:
            file_list = []
            for range in self.config.id_range:
                file_list.append(_BASE_DATA_URL_FORMAT_STR.format(
                    name=self.config.name, range=range))
        else:
            raise NotImplementedError(
                "Unknown configuration in _split_generators")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": dl_manager.download(sorted(file_list)),
                },
            )
        ]

    def _generate_examples(self, filepaths):
        if self.config.name in _CONFIG_NAMES:
            idx = 0
            for filepath in filepaths:
                with open(filepath, "rb") as f:
                    items = ijson.items(f, "item", use_float=True)
                    for data in items:
                        yield idx, data
                        idx += 1
        else:
            raise NotImplementedError(
                "Unknown configuration in _generate_examples")
