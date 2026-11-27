(() => {
    const toggle = document.querySelector('.sidebar-toggle');
    const backdrop = document.querySelector('.sidebar-backdrop');
    const sidebar = document.getElementById('app-sidebar');

    if (!toggle || !backdrop || !sidebar) {
        return;
    }

    const mobileNavigation = window.matchMedia('(max-width: 700px)');
    const setSidebarOpen = (isOpen, restoreFocus) => {
        const shouldOpen = mobileNavigation.matches && isOpen;
        document.body.classList.toggle('sidebar-open', shouldOpen);
        toggle.setAttribute('aria-expanded', String(shouldOpen));
        toggle.setAttribute('aria-label', shouldOpen ? 'Tutup navigasi' : 'Buka navigasi');
        backdrop.classList.toggle('is-visible', shouldOpen);
        backdrop.tabIndex = shouldOpen ? 0 : -1;
        sidebar.inert = mobileNavigation.matches && !shouldOpen;

        if (restoreFocus && mobileNavigation.matches && !shouldOpen) {
            toggle.focus();
        }
    };

    toggle.addEventListener('click', () => {
        setSidebarOpen(toggle.getAttribute('aria-expanded') !== 'true', false);
    });

    backdrop.addEventListener('click', () => setSidebarOpen(false, true));

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
            setSidebarOpen(false, true);
        }
    });

    const workspaceToggle = document.querySelector('.topbar-profile');
    const workspaceMenu = document.getElementById('workspace-menu');
    const workspaceMenuWrap = workspaceToggle
        ? workspaceToggle.closest('.workspace-menu-wrap')
        : null;
    if (workspaceToggle && workspaceMenu && workspaceMenuWrap) {
        const setWorkspaceMenuOpen = (isOpen, restoreFocus = false) => {
            workspaceMenu.hidden = !isOpen;
            workspaceToggle.setAttribute('aria-expanded', String(isOpen));
            if (restoreFocus) {
                workspaceToggle.focus();
            }
        };

        workspaceToggle.addEventListener('click', () => {
            setWorkspaceMenuOpen(workspaceMenu.hidden);
        });

        document.addEventListener('click', (event) => {
            if (!workspaceMenuWrap.contains(event.target)) {
                setWorkspaceMenuOpen(false);
            }
        });

        workspaceMenuWrap.addEventListener('focusout', (event) => {
            if (!workspaceMenuWrap.contains(event.relatedTarget)) {
                setWorkspaceMenuOpen(false);
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && !workspaceMenu.hidden) {
                setWorkspaceMenuOpen(false, true);
            }
        });
    }

    mobileNavigation.addEventListener('change', () => {
        setSidebarOpen(false, false);
    });
    setSidebarOpen(false, false);

    if (window.Chart) {
        const dashboardDataElement = document.getElementById('dashboard-chart-data');
        const dashboardData = dashboardDataElement
            ? JSON.parse(dashboardDataElement.textContent)
            : null;
        const dashboardChartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const value = Number(context.parsed.y).toLocaleString(
                                'id-ID',
                                { maximumFractionDigits: 3 }
                            );
                            return `${context.dataset.label}: ${value} kW`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        autoSkip: true,
                        maxTicksLimit: 8,
                        maxRotation: 0,
                    },
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#f0f2f5', drawTicks: false },
                    border: { display: false },
                    ticks: {
                        callback: (value) => `${Number(value).toLocaleString('id-ID')} kW`,
                    },
                },
            },
        };

        const consumptionCanvas = document.getElementById('consumption-chart');
        if (consumptionCanvas && dashboardData) {
            new window.Chart(consumptionCanvas, {
                type: 'line',
                data: {
                    labels: dashboardData.monthly_labels,
                    datasets: [
                        {
                            label: 'Daya aktif rata-rata',
                            data: dashboardData.monthly_power_kw,
                            borderColor: '#16a36a',
                            backgroundColor: 'rgba(22, 163, 106, 0.1)',
                            borderWidth: 2,
                            pointRadius: 2,
                            pointHoverRadius: 4,
                            fill: true,
                            tension: 0.28,
                        },
                    ],
                },
                options: dashboardChartOptions,
            });
        }

        const timeUsageCanvas = document.getElementById('time-usage-chart');
        if (timeUsageCanvas && dashboardData) {
            new window.Chart(timeUsageCanvas, {
                type: 'bar',
                data: {
                    labels: dashboardData.hourly_labels,
                    datasets: [
                        {
                            label: 'Rata-rata daya aktif',
                            data: dashboardData.hourly_power_kw,
                            backgroundColor: 'rgba(82, 118, 209, 0.7)',
                            borderRadius: 3,
                            maxBarThickness: 18,
                        },
                    ],
                },
                options: dashboardChartOptions,
            });
        }

        const forecastCanvas = document.getElementById('forecast-chart');
        if (forecastCanvas) {
            const forecastDataElement = document.getElementById('forecast-chart-data');
            const forecastData = forecastDataElement
                ? JSON.parse(forecastDataElement.textContent)
                : null;
            new window.Chart(forecastCanvas, {
                type: 'line',
                data: {
                    labels: forecastData ? forecastData.labels : [],
                    datasets: [
                        {
                            label: 'Historis',
                            data: forecastData ? forecastData.history : [],
                            borderColor: '#16a36a',
                            backgroundColor: 'rgba(22, 163, 106, 0.08)',
                            borderWidth: 2,
                            pointRadius: 1.5,
                            tension: 0.25,
                        },
                        {
                            label: 'Prediksi',
                            data: forecastData ? forecastData.prediction : [],
                            borderColor: '#7487b4',
                            borderDash: [5, 5],
                            backgroundColor: 'rgba(116, 135, 180, 0.06)',
                            borderWidth: 2,
                            pointRadius: 2,
                            tension: 0.25,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            enabled: true,
                            callbacks: {
                                label: (context) => {
                                    const value = Number(context.parsed.y).toLocaleString(
                                        'id-ID',
                                        { maximumFractionDigits: 3 }
                                    );
                                    return `${context.dataset.label}: ${value} kW`;
                                },
                            },
                        },
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                autoSkip: true,
                                maxTicksLimit: 8,
                                maxRotation: 0,
                            },
                        },
                        y: {
                            beginAtZero: true,
                            grid: { color: '#f0f2f5', drawTicks: false },
                            border: { display: false },
                            ticks: {
                                callback: (value) => `${value} kW`,
                            },
                        },
                    },
                },
            });
        }
    }

    const historyTable = document.querySelector('.history-table');
    if (historyTable) {
        const rows = Array.from(historyTable.querySelectorAll('[data-history-row]'));
        const searchInput = document.querySelector('[data-history-search]');
        const typeFilter = document.querySelector('[data-history-type]');
        const fromDate = document.querySelector('[data-history-from]');
        const toDate = document.querySelector('[data-history-to]');
        const resetButton = document.querySelector('[data-history-reset]');
        const noResultsRow = historyTable.querySelector('[data-history-no-results]');
        const recordCount = document.querySelector('[data-history-count]');

        const applyHistoryFilters = () => {
            const query = searchInput.value.trim().toLocaleLowerCase();
            const selectedType = typeFilter.value;
            const startDate = fromDate.value;
            const endDate = toDate.value;
            let visibleCount = 0;

            rows.forEach((row) => {
                const rowDate = row.dataset.date || '';
                const matches = row.textContent.toLocaleLowerCase().includes(query)
                    && (!selectedType || row.dataset.type === selectedType)
                    && (!startDate || rowDate >= startDate)
                    && (!endDate || rowDate <= endDate);

                row.hidden = !matches;
                if (matches) {
                    visibleCount += 1;
                }
            });

            noResultsRow.hidden = rows.length === 0 || visibleCount > 0;
            recordCount.textContent = String(visibleCount);
        };

        searchInput.addEventListener('input', applyHistoryFilters);
        typeFilter.addEventListener('change', applyHistoryFilters);
        fromDate.addEventListener('change', applyHistoryFilters);
        toDate.addEventListener('change', applyHistoryFilters);
        resetButton.addEventListener('click', () => {
            searchInput.value = '';
            typeFilter.value = '';
            fromDate.value = '';
            toDate.value = '';
            applyHistoryFilters();
        });
    }
})();
