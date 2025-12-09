/* --- Submit Logic --- */
document.getElementById('submitBtn').addEventListener('click', async () => {
    const btn = document.getElementById('submitBtn');
    const originalText = btn.textContent;
    btn.textContent = "Processing AI Models...";
    btn.disabled = true;

    // 1. Survey Data
    const surveyData = {
        year: document.getElementById('studentYear').value,
        major: document.getElementById('major').value
    };

    // 2. Preferences (Work Windows)
    const preferences = {
        weekdayStart: document.getElementById('weekdayStart').value,
        weekdayEnd: document.getElementById('weekdayEnd').value,
        weekendStart: document.getElementById('weekendStart').value,
        weekendEnd: document.getElementById('weekendEnd').value
    };

    // 3. Manual Courses
    const courses = [];
    document.querySelectorAll('.dynamic-card').forEach(card => {
        courses.push({
            name: card.querySelector('.assign-name').value,
            type: card.querySelector('.assign-type').value,
            date: card.querySelector('.assign-date').value
        });
    });

    const fullJson = { survey: surveyData, courses: courses, preferences: preferences };

    // 4. Build FormData
    const formData = new FormData();
    formData.append('data', JSON.stringify(fullJson));

    const pdfInput = document.getElementById('pdfUpload');
    if(pdfInput.files.length > 0) {
        for(let i=0; i<pdfInput.files.length; i++) {
            formData.append('pdfs', pdfInput.files[i]);
        }
    }

    const icsInput = document.getElementById('icsUpload');
    if(icsInput.files.length > 0) {
        formData.append('ics', icsInput.files[0]);
    }

    try {
        const response = await fetch('/api/generate-schedule', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();

        if (response.ok) {
            btn.textContent = "Complete!";
            
            // Show Download Button
            if (result.ics_url) {
                const area = document.getElementById('resultArea');
                const link = document.getElementById('downloadLink');
                area.style.display = 'block';
                link.href = result.ics_url;
                link.download = "My_Study_Schedule.ics";
            }
            
            alert(`Optimized ${result.courses.length} assignments!`);
        } else {
            alert("Error: " + (result.error || "Unknown"));
        }
    } catch (e) {
        console.error(e);
        alert("Request Failed");
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
});
