const express = require('express');
const router = express.Router();
const studentController = require('../controllers/studentController');

// Create student
router.post('/', studentController.createStudent);

// Get all students
router.get('/', studentController.getAllStudents);

// Search students
router.get('/search', studentController.searchStudents);

// Get student by ID
router.get('/:id', studentController.getStudentById);

// Update student
router.put('/:id', studentController.updateStudent);

// Update academic data only
router.patch('/:id/academic', studentController.updateAcademicData);

// Delete student
router.delete('/:id', studentController.deleteStudent);

module.exports = router;
