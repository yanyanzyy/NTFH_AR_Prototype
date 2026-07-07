using UnityEngine;
using UnityEngine.EventSystems;

namespace ARArmDetection.Facilitator
{
    public class FacilitatorPanelDragHandle : MonoBehaviour,
        IBeginDragHandler, IDragHandler, IEndDragHandler
    {
        private FacilitatorModeController _controller;

        public void Initialize(FacilitatorModeController controller)
        {
            _controller = controller;
        }

        public void OnBeginDrag(PointerEventData eventData)
        {
            _controller?.BeginPanelDrag(eventData);
        }

        public void OnDrag(PointerEventData eventData)
        {
            _controller?.DragPanel(eventData);
        }

        public void OnEndDrag(PointerEventData eventData)
        {
            _controller?.EndPanelDrag();
        }
    }
}
