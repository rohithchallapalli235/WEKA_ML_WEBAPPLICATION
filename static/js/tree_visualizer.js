/**
 * Decision Structure Visualizer JS
 * Strictly modeled after Reference Image #1:
 * Clean white canvas with colorful class node boxes (Orange, Green, Purple),
 * samples/value/class multi-line text, True/False arrow edges,
 * and automatic responsive screen fitting.
 */

let currentSvgScale = 1.0;

function renderWekaTree(treeData, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !treeData || !treeData.structure) return;

    container.innerHTML = '';

    const root = treeData.structure;

    const nodeWidth = 240;
    const nodeHeight = 96;
    const levelHeight = 160;

    let maxX = 0;
    let maxY = 0;

    function computeLeafCounts(node) {
        if (!node.children || node.children.length === 0) {
            node.leaf_count = 1;
            return 1;
        }
        node.leaf_count = node.children.reduce((sum, child) => sum + computeLeafCounts(child), 0);
        return node.leaf_count;
    }

    function layoutNode(node, depth = 0, left = 0, right = 1000) {
        node.y = 40 + depth * levelHeight;
        if (!node.children || node.children.length === 0) {
            node.x = (left + right) / 2;
        } else {
            let cursor = left;
            node.children.forEach(child => {
                const span = ((child.leaf_count || 1) / node.leaf_count) * (right - left);
                layoutNode(child, depth + 1, cursor, cursor + span);
                cursor += span;
            });
            node.x = (node.children[0].x + node.children[node.children.length - 1].x) / 2;
        }

        if (node.x > maxX) maxX = node.x;
        if (node.y > maxY) maxY = node.y;
    }

    computeLeafCounts(root);
    const totalWidth = Math.max(1100, (root.leaf_count || 4) * 260);
    layoutNode(root, 0, 40, totalWidth - 40);

    const svgWidth = totalWidth + 80;
    const svgHeight = maxY + 100;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("id", "decisionTreeSvg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMin meet");

    // Definitions for Arrow Markers
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    marker.setAttribute("id", "arrowhead");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "8");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("orient", "auto");
    marker.innerHTML = `<path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>`;
    defs.appendChild(marker);
    svg.appendChild(defs);

    const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");

    function drawNode(node) {
        // Draw Edges to Children
        if (node.children && node.children.length > 0) {
            node.children.forEach(child => {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", node.x);
                line.setAttribute("y1", node.y + nodeHeight / 2);
                line.setAttribute("x2", child.x);
                line.setAttribute("y2", child.y - nodeHeight / 2);
                line.setAttribute("stroke", "#334155");
                line.setAttribute("stroke-width", "1.8");
                line.setAttribute("marker-end", "url(#arrowhead)");
                edgeGroup.appendChild(line);

                const labelParts = [];
                if (child.edge_label) labelParts.push(child.edge_label);
                if (child.edge_sub) labelParts.push(child.edge_sub);
                const labelText = labelParts.join(' ');
                const midX = (node.x + child.x) / 2;
                const midY = (node.y + child.y) / 2;

                const edgeTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
                edgeTxt.setAttribute("x", midX);
                edgeTxt.setAttribute("y", midY - 8);
                edgeTxt.setAttribute("text-anchor", "middle");
                edgeTxt.setAttribute("font-size", "11");
                edgeTxt.setAttribute("font-family", "sans-serif");
                edgeTxt.setAttribute("font-weight", "600");
                edgeTxt.setAttribute("fill", "#334155");
                edgeTxt.textContent = labelText || "branch";
                edgeGroup.appendChild(edgeTxt);

                drawNode(child);
            });
        }

        // Draw Current Node Box (Matching Image #1)
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", node.x - nodeWidth / 2);
        rect.setAttribute("y", node.y - nodeHeight / 2);
        rect.setAttribute("width", nodeWidth);
        rect.setAttribute("height", nodeHeight);
        rect.setAttribute("rx", "6");
        rect.setAttribute("ry", "6");

        // Fill color: Leaf / pure nodes get class color, internal mixed nodes get white
        const isLeafOrPure = (node.type === 'leaf' || node.is_pure);
        const bgColor = isLeafOrPure ? (node.color || "#e28743") : "#ffffff";
        const textColor = isLeafOrPure ? "#ffffff" : "#0f172a";
        const borderColor = isLeafOrPure ? "#1e293b" : "#334155";

        rect.setAttribute("fill", bgColor);
        rect.setAttribute("stroke", borderColor);
        rect.setAttribute("stroke-width", "1.5");
        rect.setAttribute("filter", "drop-shadow(0px 2px 4px rgba(0,0,0,0.1))");
        g.appendChild(rect);

        // Text lines inside node box (Matching Image #1)
        const textGroup = document.createElementNS("http://www.w3.org/2000/svg", "text");
        textGroup.setAttribute("x", node.x);
        textGroup.setAttribute("y", node.y - 18);
        textGroup.setAttribute("text-anchor", "middle");
        textGroup.setAttribute("font-family", "sans-serif");
        textGroup.setAttribute("font-size", "10.5");
        textGroup.setAttribute("fill", textColor);

        let lineOffset = 0;

        // Line 1: Condition (for internal nodes)
        if (node.condition) {
            const tspanCond = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
            tspanCond.setAttribute("x", node.x);
            tspanCond.setAttribute("dy", "0");
            tspanCond.setAttribute("font-weight", "600");
            tspanCond.textContent = node.condition.length > 28 ? node.condition.substring(0, 26) + '...' : node.condition;
            textGroup.appendChild(tspanCond);
            lineOffset += 13;
        }

        // Line 2: samples = N
        const tspanSamples = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
        tspanSamples.setAttribute("x", node.x);
        tspanSamples.setAttribute("dy", lineOffset > 0 ? "13" : "0");
        tspanSamples.textContent = `samples = ${node.samples}`;
        textGroup.appendChild(tspanSamples);

        // Line 3: value = [...]
        const valStr = node.value ? `value = [${node.value.join(', ')}]` : '';
        if (valStr) {
            const tspanVal = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
            tspanVal.setAttribute("x", node.x);
            tspanVal.setAttribute("dy", "13");
            tspanVal.textContent = valStr.length > 26 ? valStr.substring(0, 24) + '...]' : valStr;
            textGroup.appendChild(tspanVal);
        }

        // Line 4: class = X
        if (node.class) {
            const tspanClass = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
            tspanClass.setAttribute("x", node.x);
            tspanClass.setAttribute("dy", "13");
            tspanClass.setAttribute("font-weight", "700");
            tspanClass.textContent = `class = ${node.class}`;
            textGroup.appendChild(tspanClass);
        }

        g.appendChild(textGroup);
        nodeGroup.appendChild(g);
    }

    drawNode(root);

    svg.appendChild(edgeGroup);
    svg.appendChild(nodeGroup);
    container.appendChild(svg);
}

function zoomTree(delta) {
    const svg = document.getElementById("decisionTreeSvg");
    if (!svg) return;
    currentSvgScale = Math.max(0.4, Math.min(2.5, currentSvgScale + delta));
    svg.style.transform = `scale(${currentSvgScale})`;
    svg.style.transformOrigin = "top center";
}

function fitTreeToScreen() {
    const svg = document.getElementById("decisionTreeSvg");
    if (!svg) return;
    currentSvgScale = 1.0;
    svg.style.transform = "scale(1.0)";
    svg.style.transformOrigin = "top center";
}
