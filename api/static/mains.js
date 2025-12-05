let rowCount = 1;

// Drag and drop functionality
const rankingContainer = document.getElementById('rankingContainer');
let draggedItem = null;

rankingContainer.addEventListener('dragstart', (e) => {
    if (e.target.classList.contains('ranking-item')) {
        draggedItem = e.target;
        e.target.classList.add('dragging');
    }
});

rankingContainer.addEventListener('dragend', (e) => {
    if (e.target.classList.contains('ranking-item')) {
        e.target.classList.remove('dragging');
    }
});

rankingContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    const afterElement = getDragAfterElement(rankingContainer, e.clientY);
    if (afterElement == null) {
        rankingContainer.appendChild(draggedItem);
    } else {
        rankingContainer.insertBefore(draggedItem, afterElement);
    }
});

function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.ranking-item:not(.dragging)')];
    
    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        
        if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
        } else {
            return closest;
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

// Table functions
function addRow() {
    rowCount++;
    const tableBody = document.getElementById('courseTableBody');
    const newRow = document.createElement('tr');
    
    newRow.innerHTML = `
        <td>${rowCount}</td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="assignment">Assignment</option>
                <option value="writing">Writing</option>
                <option value="lab">Lab</option>
                <option value="project">Project</option>
                <option value="others">Others</option>
            </select>
        </td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="business">Business</option>
                <option value="stem">STEM</option>
                <option value="humanities">Humanities</option>
                <option value="arts">Arts</option>
                <option value="others">Others</option>
            </select>
        </td>
        <td><input type="text" placeholder=""></td>
        <td><input type="text" placeholder=""></td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="1-2">1-2</option>
                <option value="3-5">3-5</option>
                <option value="6-10">6-10</option>
                <option value="10+">10+</option>
            </select>
        </td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="textbook">Textbook</option>
                <option value="google">Google</option>
                <option value="ai">AI</option>
                <option value="mixed">Mixed</option>
            </select>
        </td>
        <td>
            <button class="delete-btn" onclick="deleteRow(this)">Delete</button>
        </td>
    `;
    
    tableBody.appendChild(newRow);
}

function deleteRow(button) {
    const row = button.parentElement.parentElement;
    row.remove();
    updateRowNumbers();
}

function updateRowNumbers() {
    const rows = document.querySelectorAll('#courseTableBody tr');
    rowCount = rows.length;
    rows.forEach((row, index) => {
        row.cells[0].textContent = index + 1;
    });
}

// Collect form data
function collectFormData() {
    const surveyData = {
        year: document.getElementById('year').value,
        major: document.getElementById('major').value,
        workInGroup: document.getElementById('workInGroup').value,
        workLocation: document.getElementById('workLocation').value,
        preferredWorkingTime: getWorkingTimeRanking()
    };

    const courses = [];
    const rows = document.querySelectorAll('#courseTableBody tr');
    rows.forEach((row, index) => {
        const cells = row.cells;
        const course = {
            number: index + 1,
            type: cells[1].querySelector('select').value,
            classMajor: cells[2].querySelector('select').value,
            className: cells[3].querySelector('input').value,
            assignmentName: cells[4].querySelector('input').value,
            sessions: cells[5].querySelector('select').value,
            resources: cells[6].querySelector('select').value
        };
        courses.push(course);
    });

    return {
        survey: surveyData,
        courses: courses
    };
}

function getWorkingTimeRanking() {
    const rankingItems = document.querySelectorAll('#rankingContainer .ranking-item');
    const ranking = [];
    rankingItems.forEach((item, index) => {
        ranking.push({
            rank: index + 1,
            time: item.getAttribute('data-time')
        });
    });
    return ranking;
}

// Submit handler
document.querySelector('.submit-btn').addEventListener('click', async function() {
    const button = this;
    const originalText = button.textContent;
    button.textContent = 'Generating... ⏳';
    button.disabled = true;

    try {
        const formData = new FormData();
        const jsonData = collectFormData();
        formData.append('data', JSON.stringify(jsonData));

        // Add PDFs
        const pdfFiles = document.getElementById('pdfUpload').files;
        for (let i = 0; i < pdfFiles.length; i++) {
            formData.append('pdfs', pdfFiles[i]);
        }

        // Add ICS
        const icsFile = document.getElementById('icsUpload').files[0];
        if (icsFile) {
            formData.append('ics', icsFile);
        }

        // Send to backend
        const response = await fetch('/api/generate-schedule', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Server responded with: ' + response.status);
        }

        const result = await response.json();
        
        // Show success message
        let message = '✅ Schedule Generated Successfully!\n\n';
        message += `📊 Survey Data Received\n`;
        message += `📚 Courses Processed: ${result.courses_count}\n`;
        message += `📄 PDFs Analyzed: ${result.pdfs_processed}\n`;
        message += `📅 Calendar Events: ${result.calendar_events_count}\n`;
        
        alert(message);
        
        console.log('Full Response:', result);
        
        button.textContent = originalText;
        button.disabled = false;

    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error: ' + error.message + '\n\nMake sure the Flask server is running on port 5001');
        button.textContent = originalText;
        button.disabled = false;
    }
});
