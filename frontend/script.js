document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('addEncodingColumn').addEventListener('click', function() {
        const container = document.getElementById('encodingColumnsContainer');
        const newEntry = document.createElement('div');
        newEntry.className = 'encoding-entry';

        const input = document.createElement('input');
        input.type = 'text';
        input.name = 'encodingColumn';
        input.placeholder = 'Column Name';

        const select = document.createElement('select');
        select.name = 'encodingType';

        const ordinalOption = document.createElement('option');
        ordinalOption.value = 'ordinal';
        ordinalOption.text = 'Ordinal Encoding';

        const looOption = document.createElement('option');
        looOption.value = 'loo';
        looOption.text = 'Leave One Out Encoding';

        const labelOption = document.createElement('option');
        labelOption.value = 'label';
        labelOption.text = 'Label Encoding';

        select.appendChild(ordinalOption);
        select.appendChild(looOption);
        select.appendChild(labelOption);

        newEntry.appendChild(input);
        newEntry.appendChild(select);
        container.appendChild(newEntry);
    });

    document.getElementById('tableUpload').addEventListener('change', function() {
        const file = this.files[0];
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                document.getElementById('dataframeHead').innerHTML = data.dataframe_head;
                const correlationHeatmap = document.getElementById('correlationHeatmap');
                correlationHeatmap.src = `data:image/jpeg;base64,${data.correlationHeatmap}`;
                document.getElementById('runRF').dataset.filepath = data.filepath;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    document.getElementById('runRF').addEventListener('click', function() {
        // Reset progress bar
        const progressBar = document.getElementById('progressBar');
        progressBar.style.width = '0%';
        progressBar.innerText = '0%';

        // Retrieve the file path from the dataset
        const filePath = this.dataset.filepath;
        const targetColumn = document.getElementById('targetColumn').value;
        const taskType = document.getElementById('dropdown').value;
        const encodingEntries = document.querySelectorAll('.encoding-entry');
        const encodingColumns = {};

        encodingEntries.forEach(entry => {
            const colName = entry.querySelector('input[name="encodingColumn"]').value;
            const encType = entry.querySelector('select[name="encodingType"]').value;
            encodingColumns[colName] = encType;
        });

        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filepath: filePath,
                target_column: targetColumn,
                encoding_columns: encodingColumns,
                task_type: taskType
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                alert(data.message);
                // Store the model path in the predictRF button's dataset
                document.getElementById('predictRF').dataset.modelPath = data.model_path;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    document.getElementById('predictTableUpload').addEventListener('change', function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = function(e) {
            const contents = e.target.result;
            const tablePreview = document.getElementById('dataframeTest');

            // Clear previous contents
            tablePreview.innerHTML = '';

            const rows = contents.split('\n');
            const table = document.createElement('table');
            const tbody = document.createElement('tbody');

            rows.forEach((row, rowIndex) => {
                const tr = document.createElement('tr');
                const cells = row.split(',');

                cells.forEach((cell, cellIndex) => {
                    const td = document.createElement(rowIndex === 0 ? 'th' : 'td');
                    td.innerText = cell;
                    tr.appendChild(td);
                });

                tbody.appendChild(tr);
            });

            table.appendChild(tbody);
            tablePreview.appendChild(table);
        };

        reader.readAsText(file);
    });

    document.getElementById('predictRF').addEventListener('click', function() {
        const fileInput = document.getElementById('predictTableUpload');
        const file = fileInput.files[0];
        const formData = new FormData();

        formData.append('file', file);
        formData.append('task_type', document.getElementById('dropdown').value);
        formData.append('model_path', document.getElementById('predictRF').dataset.modelPath);
        formData.append('target_column', document.getElementById('targetColumn').value);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                // Display the prediction result
                document.getElementById('targetPredict').innerText = data.prediction;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    const socket = io();

    socket.on('progress', (data) => {
        const progressBar = document.getElementById('progressBar');
        progressBar.style.width = `${data.progress}%`;
        progressBar.innerText = `${data.progress.toFixed(2)}%`;
    });

    socket.on('train_complete', (data) => {
        alert('Training Complete!');
        const featureImportanceImg = document.getElementById('featureImportance');
        featureImportanceImg.src = `data:image/png;base64,${data.importance}`;
        document.getElementById('performanceMetrics').innerHTML = data.metrics;
    });
});

