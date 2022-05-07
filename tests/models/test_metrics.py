import lzma
import os
import csv
import shutil
from tempfile import mkdtemp
import logging

from deeprank_gnn.models.metrics import (MetricsExporterCollection,
                                         TensorboardBinaryClassificationExporter,
                                         OutputExporter)

_log = logging.getLogger(__name__)

def test_metrics_export():

    tmp_dir = mkdtemp()
    try:
        output_exporter = OutputExporter(tmp_dir)
        exporters = [TensorboardBinaryClassificationExporter(tmp_dir),
                     output_exporter]

        collection = MetricsExporterCollection(*exporters)

        pass_name = "test"
        epoch_number = 0

        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]

        with collection:
            collection.process(pass_name, epoch_number,
                               entry_names,
                               outputs,
                               targets)

        with lzma.open(output_exporter.get_filename(pass_name, epoch_number), 'rt', newline='\n') as table_file:
            r = csv.reader(table_file, delimiter=',')
            header = next(r)
            columns = {name: [] for name in header}
            for row in r:
                for column_index, column_name in enumerate(header):
                    columns[column_name].append(row[column_index])

        assert columns["entry"] == entry_names, f"{columns['entry']} != {entry_names}"
        assert columns["output"] == [str(z) for z in outputs], f"columns['output'] != {outputs}"
        assert columns["target"] == [str(y) for y in targets], f"columns['target'] != {targets}"

        assert len(os.listdir(tmp_dir)) == 2  # tensorboard & table
    finally:
        shutil.rmtree(tmp_dir)
