using System;
using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection.Facilitator
{
    public enum FacilitatorAdvanceMode
    {
        Manual,
        AfterNarration,
        ExternalSignal,
    }

    [Serializable]
    public class FacilitatorStep
    {
        [SerializeField] private string _id;
        [SerializeField] private string _title;
        [TextArea(3, 7)] [SerializeField] private string _instruction;
        [SerializeField] private AudioClip _narration;
        [SerializeField] private FacilitatorAdvanceMode _advanceMode;
        [SerializeField] private string _completionSignal;

        public string Id => _id;
        public string Title => _title;
        public string Instruction => _instruction;
        public AudioClip Narration => _narration;
        public FacilitatorAdvanceMode AdvanceMode => _advanceMode;
        public string CompletionSignal => _completionSignal;
    }

    [CreateAssetMenu(fileName = "FacilitatorProcedure", menuName = "NTFH/Facilitator Procedure")]
    public class FacilitatorProcedure : ScriptableObject
    {
        [SerializeField] private string _procedureTitle = "Venepuncture - Vacutainer Method";
        [SerializeField] private List<FacilitatorStep> _steps = new();

        public string ProcedureTitle => _procedureTitle;
        public IReadOnlyList<FacilitatorStep> Steps => _steps;
    }
}
