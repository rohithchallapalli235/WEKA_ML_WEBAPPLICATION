/**
 * ROC Curve Visualizer JS
 * Renders multi-class ROC curve canvas matching reference screenshots
 */

function renderRocCanvas(rocData, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !rocData || rocData.length === 0) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width = 540;
    const height = canvas.height = 360;

    // Margins for axes
    const margin = { top: 40, right: 140, bottom: 50, left: 60 };
    const graphWidth = width - margin.left - margin.right;
    const graphHeight = height - margin.top - margin.bottom;

    // Clear canvas
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Border
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(margin.left, margin.top, graphWidth, graphHeight);

    // Title
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("ROC Curve", margin.left + graphWidth / 2, margin.top - 15);

    // Axis Labels
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#334155";
    ctx.fillText("False Positive Rate", margin.left + graphWidth / 2, height - 12);

    ctx.save();
    ctx.translate(18, margin.top + graphHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("True Positive Rate", 0, 0);
    ctx.restore();

    // Grid ticks (0.0 to 1.0)
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {
        const val = (i / 5).toFixed(1);
        const x = margin.left + (i / 5) * graphWidth;
        const y = margin.top + graphHeight - (i / 5) * graphHeight;

        // X tick
        ctx.textAlign = "center";
        ctx.fillText(val, x, margin.top + graphHeight + 15);

        // Y tick
        ctx.textAlign = "right";
        ctx.fillText(val, margin.left - 8, y + 4);

        // Grid lines
        if (i > 0 && i < 5) {
            ctx.beginPath();
            ctx.moveTo(x, margin.top);
            ctx.lineTo(x, margin.top + graphHeight);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(margin.left + graphWidth, y);
            ctx.stroke();
        }
    }

    // Diagonal dashed line (0,0) -> (1,1)
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + graphHeight);
    ctx.lineTo(margin.left + graphWidth, margin.top);
    ctx.stroke();
    ctx.restore();

    // Palette for multi-class ROC curves
    const palette = ["#0284c7", "#d97706", "#16a34a", "#dc2626", "#9333ea", "#0891b2"];

    // Draw ROC Curves & Legend
    const legendX = margin.left + graphWidth + 12;
    let legendY = margin.top + 15;

    rocData.forEach((curve, idx) => {
        const color = palette[idx % palette.length];
        const fpr = curve.fpr;
        const tpr = curve.tpr;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();

        for (let j = 0; j < fpr.length; j++) {
            const px = margin.left + fpr[j] * graphWidth;
            const py = margin.top + graphHeight - tpr[j] * graphHeight;
            if (j === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Legend item
        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY - 8, 12, 3);

        ctx.fillStyle = "#1e293b";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "left";
        const labelText = `${curve.class} (AUC=${curve.auc.toFixed(2)})`;
        ctx.fillText(labelText, legendX + 18, legendY - 5);

        legendY += 20;
    });
}
