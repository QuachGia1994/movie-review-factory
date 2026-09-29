# Workspace UX repair

Product direction: [Creator product identity](../biz/creator-product.md). This plan corrects the first-run/dashboard layout after the horizontal-toolbar redesign created overlapping controls and large dead space.

## Target
A solo creator should see one clear primary action before a project exists, then a stable project workspace after selection. Global tools must never compete for page layout.

## Changes
1. Replace modal tasks in the toolbar with explicit dialog actions: create project, brand/settings, and library/series.
2. Keep only the project switcher as an anchored compact menu; show the active project in its trigger.
3. Merge cross-project search and project/series management under one Library dialog.
4. Turn the empty canvas into a first-run start state with one primary “Tạo project” action and an “Mở project” secondary action when projects exist.
5. Preserve all existing element IDs/API handlers and approval gates while changing container structure.
6. Consolidate dashboard-shell/top-toolbar CSS so one layout definition owns desktop/mobile behavior; remove competing absolute panel rules.
7. Verify first-run, existing-project selection, modal dismissal, 1790×850 desktop, narrower viewport, pytest, and diff hygiene.

## Acceptance
- No control panel overlaps another control panel.
- Empty state does not leave a left-side stacked toolbar or accidental form overlay.
- Create form is not dismissed by clicking elsewhere; it closes only by explicit close or successful project creation.
- Brand and Library tools open independently in bounded dialogs.
- Project switcher is compact, scrolls internally, and closes on outside click/Escape.
- No horizontal page overflow at desktop or narrow viewport.
- Existing backend behavior and script approval gate remain unchanged.

Status: active · 2026-09-28.
